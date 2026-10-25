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

# make sure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer
from src.models.shap_explainer import explain as shap_explain
from src.models.evaluation import cross_validate_models, temporal_evaluation


def main(args):
    # ── 1. Load data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION PIPELINE  (3 models + SHAP)")
    print("=" * 60)

    loader = DataLoader(args.data_path)
    df = loader.load_data()

    # ── 2. Split ───────────────────────────────────────────
    train_df, val_df, test_df = loader.split(df)

    # ── 3. Cross-validation on training data only ──────────
    trainer = ModelTrainer(experiment_name=args.experiment)
    cv_results = cross_validate_models(
        train_df, trainer._build_model_specs, n_splits=5, seed=42
    )
    temporal_results = temporal_evaluation(
        df, trainer._build_model_specs
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
    results = trainer.train(X_train, y_train, X_val, y_val)

    # ── 8. Evaluate on test set ────────────────────────────
    test_metrics = trainer.evaluate_test(X_test, y_test)

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