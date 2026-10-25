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
        "temporal_cross_split_overlap": overlap_counts(temporal_parts),
        "policy": (
            "Retain duplicates: repeated transactions may be legitimate events. "
            "Report overlap rather than silently dropping rows."
        ),
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
