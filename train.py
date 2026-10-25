"""
Main training script — runs the full pipeline from data loading
through model evaluation and saving.

Usage:
    python train.py                    # full pipeline
    python train.py                    # feature engineering is required
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost

# make sure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer
from src.models.shap_explainer import explain as shap_explain
from src.models.evaluation import (
    cross_validate_models,
    duplicate_audit,
    duplicate_sensitivity_evaluation,
    git_revision,
    temporal_evaluation,
)


def main(args):
    # ── 1. Load data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION PIPELINE  (3 models + SHAP)")
    print("=" * 60)

    loader = DataLoader(args.data_path)
    df = loader.load_data()

    # ── 2. Split ───────────────────────────────────────────
    train_df, val_df, test_df = loader.split(df)
    raw_test_df = test_df.copy()
    raw_train_df, raw_val_df, raw_test_df_with_duplicates = loader.split(
        df, deduplicate=False
    )

    # ── 3. Cross-validation on training data only ──────────
    trainer = ModelTrainer(experiment_name=args.experiment)
    cv_results = cross_validate_models(
        train_df, trainer._build_model_specs, n_splits=5, seed=42
    )
    duplicate_sensitivity = duplicate_sensitivity_evaluation(
        df, trainer._build_model_specs
    )
    temporal_results = temporal_evaluation(df, trainer._build_model_specs)
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    temporal_cuts = [int(len(ordered) * 0.70), int(len(ordered) * 0.80)]
    temporal_parts = [
        ordered.iloc[:temporal_cuts[0]],
        ordered.iloc[temporal_cuts[0]:temporal_cuts[1]],
        ordered.iloc[temporal_cuts[1]:],
    ]
    duplicate_results = duplicate_audit(
        df,
        {
            "train": raw_train_df,
            "validation": raw_val_df,
            "test": raw_test_df_with_duplicates,
        },
        temporal_parts,
    )

    # ── 4. Feature engineering (fit on raw training data only) ──
    fe = FeatureEngineer()
    train_df = fe.fit_transform(train_df)
    val_df = fe.transform(val_df)
    test_df = fe.transform(test_df)
    print(f"Features after engineering: {train_df.shape[1] - 1}")

    # ── 5. Preprocess (fit on training data only) ──────────
    preprocessor = DataPreprocessor()
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # ── 6. Separate X and y ────────────────────────────────
    X_train, y_train = loader.get_feature_targets(train_df)
    X_val, y_val = loader.get_feature_targets(val_df)
    X_test, y_test = loader.get_feature_targets(test_df)

    # ── 7. Train models ────────────────────────────────────
    results = trainer.train(
        X_train, y_train, X_val, y_val, cv_results=cv_results
    )

    # ── 8. Evaluate on test set ────────────────────────────
    test_metrics = trainer.evaluate_test(X_test, y_test)
    test_scores = trainer.best_model.predict_proba(X_test)[:, 1]
    test_labels = y_test.to_numpy()
    test_predictions = test_scores >= trainer.best_threshold
    precision, recall, _ = __import__(
        "sklearn.metrics", fromlist=["precision_recall_curve"]
    ).precision_recall_curve(test_labels, test_scores)
    evaluation_dir = Path("outputs/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with open(evaluation_dir / "precision_recall_curve.json", "w") as f:
        json.dump(
            {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "selected_threshold": trainer.best_threshold,
            },
            f,
            indent=2,
        )
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label="Held-out test PR curve")
    selected = test_metrics["threshold_metrics"][str(
        round(trainer.best_threshold, 6)
    )]
    plt.scatter(
        selected["recall"],
        selected["precision"],
        color="red",
        label=f"Selected threshold={trainer.best_threshold:.6f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve (held-out test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(evaluation_dir / "precision_recall_curve.png", dpi=150)
    plt.close()

    example_indices = {
        "correctly_detected_fraud": np.where(
            (test_labels == 1) & test_predictions
        )[0][np.argmax(test_scores[(test_labels == 1) & test_predictions])],
        "borderline_case": int(np.argmin(
            np.abs(test_scores - trainer.best_threshold)
        )),
        "false_negative": np.where(
            (test_labels == 1) & ~test_predictions
        )[0][np.argmax(test_scores[(test_labels == 1) & ~test_predictions])],
        "false_positive": np.where(
            (test_labels == 0) & test_predictions
        )[0][np.argmax(test_scores[(test_labels == 0) & test_predictions])],
    }
    examples = {}
    for name, index in example_indices.items():
        raw = raw_test_df.iloc[index].drop(labels=["Class"], errors="ignore")
        examples[name] = {
            "test_row_index": int(index),
            "input": {key: float(value) for key, value in raw.items()},
            "true_label": int(test_labels[index]),
            "model_score": float(test_scores[index]),
            "predicted_class": int(test_predictions[index]),
            "threshold": trainer.best_threshold,
        }
    with open(evaluation_dir / "prediction_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # ── 9. Save model and preprocessor ────────────────────
    trainer.save(
        args.model_dir,
        feature_names=X_train.columns,
        cv_results=cv_results,
        temporal_results=temporal_results,
    )

    # Save the fitted preprocessor so the API can apply
    # the exact same scaling at inference time.
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    joblib.dump(fe, model_dir / "feature_engineer.pkl")
    print(f"Preprocessor saved to {model_dir}/")

    # ── 10. Save results ───────────────────────────────────
    output = {
        "validation_results": results,
        "test_results": test_metrics,
        "best_model": trainer.best_model_name,
        "failed_models": trainer.failures,
        "cross_validation": cv_results,
        "temporal_evaluation": temporal_results,
        "duplicate_audit": duplicate_results,
        "duplicate_sensitivity": duplicate_sensitivity,
        "provenance": {
            "seed": 42,
            "split_strategy": "stratified 70/10/20 with random_state=42",
            "git_commit": git_revision(),
            "library_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
        },
    }
    results_path = Path(args.model_dir) / "results.json"
    # default=str handles non-serialisable types like numpy floats
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # ── 10. SHAP interpretability (winner only) ────────────────
    shap_explain(
        model=trainer.best_model,
        model_name=trainer.best_model_name,
        X_test=X_test,
        y_test=y_test,
        output_dir="outputs/shap",
    )

    print("\nPipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fraud detection models"
    )
    parser.add_argument(
        "--data-path",
        default="data/creditcard.csv",
        help="Path to the credit card CSV",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--experiment",
        default="fraud-detection",
        help="MLflow experiment name",
    )
    args = parser.parse_args()
    main(args)