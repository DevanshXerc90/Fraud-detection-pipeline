"""Small, reproducible evaluation helpers used by the training pipeline."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.data import DataPreprocessor
from src.features import FeatureEngineer


def cross_validate_models(raw_train, model_factory, n_splits=5, seed=42):
    """Compare models with preprocessing fitted independently in each fold."""
    y = raw_train["Class"].to_numpy()
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    scores = {}
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(raw_train, y)):
        fold_train = raw_train.iloc[train_idx]
        fold_valid = raw_train.iloc[valid_idx]
        fe = FeatureEngineer()
        pp = DataPreprocessor()
        transformed_train = pp.fit_transform(fe.fit_transform(fold_train))
        transformed_valid = pp.transform(fe.transform(fold_valid))
        X_train = transformed_train.drop(columns=["Class"])
        y_train = transformed_train["Class"]
        X_valid = transformed_valid.drop(columns=["Class"])
        y_valid = transformed_valid["Class"]
        negatives = int((y_train == 0).sum())
        positives = int((y_train == 1).sum())
        for name, model in model_factory(negatives / max(positives, 1)).items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            fold_score = average_precision_score(
                y_valid, fitted.predict_proba(X_valid)[:, 1]
            )
            scores.setdefault(name, []).append(float(fold_score))
    return {
        name: {
            "fold_auprc": values,
            "mean_auprc": float(np.mean(values)),
            "std_auprc": float(np.std(values, ddof=1)),
            "folds": n_splits,
        }
        for name, values in scores.items()
    }


def temporal_evaluation(raw_data, model_factory):
    """Evaluate an earlier-train/later-validation/later-test chronology."""
    ordered = raw_data.sort_values("Time", kind="mergesort").reset_index(drop=True)
    first_cut = int(len(ordered) * 0.70)
    second_cut = int(len(ordered) * 0.80)
    train = ordered.iloc[:first_cut]
    validation = ordered.iloc[first_cut:second_cut]
    test = ordered.iloc[second_cut:]
    fe = FeatureEngineer()
    pp = DataPreprocessor()
    transformed_train = pp.fit_transform(fe.fit_transform(train))
    transformed_validation = pp.transform(fe.transform(validation))
    transformed_test = pp.transform(fe.transform(test))
    X_train = transformed_train.drop(columns=["Class"])
    y_train = transformed_train["Class"]
    X_validation = transformed_validation.drop(columns=["Class"])
    y_validation = transformed_validation["Class"]
    X_test = transformed_test.drop(columns=["Class"])
    y_test = transformed_test["Class"]
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model_results = {}
    for name, model in model_factory(ratio).items():
        fitted = clone(model).fit(X_train, y_train)
        validation_score = fitted.predict_proba(X_validation)[:, 1]
        precision, recall, thresholds = precision_recall_curve(
            y_validation, validation_score
        )
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-12
        )
        threshold = float(thresholds[int(np.argmax(f1))])
        test_score = fitted.predict_proba(X_test)[:, 1]
        confusion = confusion_matrix(
            y_test, test_score >= threshold, labels=[0, 1]
        ).tolist()
        model_results[name] = {
            "validation_auprc": float(
                average_precision_score(y_validation, validation_score)
            ),
            "test_auprc": float(average_precision_score(y_test, test_score)),
            "test_precision": float(
                precision_score(y_test, test_score >= threshold, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, test_score >= threshold, zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_test, test_score >= threshold, zero_division=0)
            ),
            "test_confusion_matrix": confusion,
            "test_bootstrap_confidence_intervals": bootstrap_confidence_intervals(
                y_test, test_score, threshold
            ),
            "threshold": threshold,
        }
    best_name = max(
        model_results, key=lambda name: model_results[name]["validation_auprc"]
    )
    return {
        "ordering": "ascending Time (elapsed seconds)",
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "fraud_counts": {
            "train": int(y_train.sum()),
            "validation": int(y_validation.sum()),
            "test": int(y_test.sum()),
        },
        "best_model": best_name,
        "models": model_results,
    }


def duplicate_audit(df, stratified_parts, temporal_parts):
    """Measure exact duplicate rows and overlap without dropping any data."""
    duplicate_mask = df.duplicated(keep=False)

    def overlap_counts(parts):
        hashes = [
            set(pd.util.hash_pandas_object(part, index=False).tolist())
            for part in parts
        ]
        return {
            f"{left}_{right}": len(hashes[left] & hashes[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        }

    stratified_hashes = {
        name: set(pd.util.hash_pandas_object(part, index=False).tolist())
        for name, part in stratified_parts.items()
    }
    all_other_hashes = {
        name: set().union(
            *(hashes for other, hashes in stratified_hashes.items() if other != name)
        )
        for name in stratified_hashes
    }
    test = stratified_parts["test"]
    test_hashes = pd.util.hash_pandas_object(test, index=False)
    test_fraud_twin_mask = (
        (test["Class"].to_numpy() == 1)
        & test_hashes.isin(
            stratified_hashes["train"] | stratified_hashes["validation"]
        ).to_numpy()
    )
    all_fraud_twin_count = 0
    for name, part in stratified_parts.items():
        hashes = pd.util.hash_pandas_object(part, index=False)
        all_fraud_twin_count += int(
            (
                (part["Class"].to_numpy() == 1)
                & hashes.isin(all_other_hashes[name]).to_numpy()
            ).sum()
        )
    return {
        "exact_duplicate_rows_in_dataset": int(duplicate_mask.sum()),
        "duplicate_groups": int(
            df.loc[duplicate_mask].drop_duplicates().shape[0]
        ),
        "duplicate_rows_beyond_first": int(df.duplicated().sum()),
        "stratified_within_split": {
            name: int(part.duplicated().sum())
            for name, part in stratified_parts.items()
        },
        "stratified_cross_split_overlap": overlap_counts(
            list(stratified_parts.values())
        ),
        "stratified_test_frauds_with_train_validation_twin": int(
            test_fraud_twin_mask.sum()
        ),
        "all_frauds_with_twin_in_another_stratified_split": all_fraud_twin_count,
        "temporal_cross_split_overlap": overlap_counts(temporal_parts),
        "policy": (
            "Retain duplicates: repeated transactions may be legitimate events. "
            "Report overlap rather than silently dropping rows."
        ),
    }


def _fit_stratified_run(df, model_factory, deduplicate):
    """Fit and evaluate one stratified policy without writing artifacts."""
    from src.data import DataLoader
    from src.models.trainer import ModelTrainer

    loader = DataLoader()
    train, validation, test = loader.split(df, deduplicate=deduplicate)
    cv = cross_validate_models(train, model_factory, n_splits=5, seed=42)
    fe = FeatureEngineer().fit(train)
    train_features = fe.transform(train)
    validation_features = fe.transform(validation)
    test_features = fe.transform(test)
    pp = DataPreprocessor().fit(train_features)
    train_features = pp.transform(train_features)
    validation_features = pp.transform(validation_features)
    test_features = pp.transform(test_features)
    X_train = train_features.drop(columns=["Class"])
    y_train = train_features["Class"]
    X_validation = validation_features.drop(columns=["Class"])
    y_validation = validation_features["Class"]
    X_test = test_features.drop(columns=["Class"])
    y_test = test_features["Class"]
    trainer = ModelTrainer(experiment_name="fraud-detection-sensitivity")
    trainer.train(X_train, y_train, X_validation, y_validation, cv_results=cv)
    scores = trainer.best_model.predict_proba(X_test)[:, 1]
    predictions = scores >= trainer.best_threshold
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return {
        "deduplicated_before_split": deduplicate,
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "fraud_counts": {
            "train": int(y_train.sum()),
            "validation": int(y_validation.sum()),
            "test": int(y_test.sum()),
        },
        "selected_model": trainer.best_model_name,
        "threshold": trainer.best_threshold,
        "auprc": float(average_precision_score(y_test, scores)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "confusion_counts": {
            "true_negatives": matrix[0][0],
            "false_positives": matrix[0][1],
            "false_negatives": matrix[1][0],
            "true_positives": matrix[1][1],
        },
        "bootstrap_confidence_intervals": bootstrap_confidence_intervals(
            y_test, scores, trainer.best_threshold
        ),
        "parts": (train, validation, test),
        "test_labels": y_test.to_numpy(),
        "test_scores": scores,
        "feature_engineer": fe,
        "preprocessor": pp,
        "test_features": X_test,
    }


def duplicate_sensitivity_evaluation(df, model_factory):
    """Compare raw, pre-split deduplicated, and clean-test-only evaluations."""
    raw = _fit_stratified_run(df, model_factory, deduplicate=False)
    deduped = _fit_stratified_run(df, model_factory, deduplicate=True)
    train, validation, test = raw["parts"]
    reference_hashes = set(
        pd.util.hash_pandas_object(
            pd.concat([train, validation]), index=False
        ).tolist()
    )
    test_hashes = pd.util.hash_pandas_object(test, index=False)
    clean_mask = ~test_hashes.isin(reference_hashes).to_numpy()
    clean_labels = raw["test_labels"][clean_mask]
    clean_scores = raw["test_scores"][clean_mask]
    clean_predictions = clean_scores >= raw["threshold"]
    clean_matrix = confusion_matrix(
        clean_labels, clean_predictions, labels=[0, 1]
    ).tolist()
    clean = {
            "deduplicated_before_split": False,
            "test_restricted_to_no_train_validation_twin": True,
            "test_rows": int(clean_mask.sum()),
            "fraud_count": int(clean_labels.sum()),
            "auprc": float(average_precision_score(clean_labels, clean_scores)),
            "precision": float(
                precision_score(clean_labels, clean_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(clean_labels, clean_predictions, zero_division=0)
            ),
            "confusion_counts": {
                "true_negatives": clean_matrix[0][0],
                "false_positives": clean_matrix[0][1],
                "false_negatives": clean_matrix[1][0],
                "true_positives": clean_matrix[1][1],
            },
            "bootstrap_confidence_intervals": bootstrap_confidence_intervals(
                clean_labels, clean_scores, raw["threshold"]
            ),
            "threshold": raw["threshold"],
        }
    return {
            "current_stratified": {
                key: value for key, value in raw.items() if key not in {
                    "parts", "test_labels", "test_scores", "feature_engineer",
                    "preprocessor", "test_features",
                }
            },
            "deduplicated_before_split": {
                key: value for key, value in deduped.items() if key not in {
                    "parts", "test_labels", "test_scores", "feature_engineer",
                    "preprocessor", "test_features",
                }
            },
            "current_split_clean_test": clean,
    }


def threshold_metrics(y_true, scores, thresholds):
    """Return precision, recall, F1, and confusion matrix per threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    return {
        str(threshold): {
            "precision": float(precision_score(
                y_true, scores >= threshold, zero_division=0
            )),
            "recall": float(recall_score(
                y_true, scores >= threshold, zero_division=0
            )),
            "f1": float(f1_score(
                y_true, scores >= threshold, zero_division=0
            )),
            "confusion_matrix": confusion_matrix(
                y_true, scores >= threshold, labels=[0, 1]
            ).tolist(),
        }
        for threshold in thresholds
    }


def bootstrap_confidence_intervals(
    y_true,
    scores,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Calculate percentile bootstrap CIs, skipping samples without positives."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = {"auprc": [], "precision": [], "recall": []}
    skipped = 0

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[indices]
        if not np.any(sampled_y == 1):
            skipped += 1
            continue
        sampled_scores = scores[indices]
        predicted = sampled_scores >= threshold
        values["auprc"].append(
            average_precision_score(sampled_y, sampled_scores)
        )
        values["precision"].append(
            precision_score(sampled_y, predicted, zero_division=0)
        )
        values["recall"].append(
            recall_score(sampled_y, predicted, zero_division=0)
        )

    if not all(values.values()):
        raise ValueError("Bootstrap produced no valid positive-containing samples")
    result = {
        metric: {
            "lower": float(np.percentile(metric_values, 2.5)),
            "upper": float(np.percentile(metric_values, 97.5)),
            "replicates": len(metric_values),
        }
        for metric, metric_values in values.items()
    }
    result.update({
        "method": "percentile bootstrap, 95% CI",
        "requested_replicates": n_bootstrap,
        "skipped_without_positive": skipped,
        "seed": seed,
    })
    return result


def utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def git_revision() -> str:
    """Return the current commit without failing outside a Git clone."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
