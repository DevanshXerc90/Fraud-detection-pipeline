"""
Main training script — runs the full pipeline from data loading
through model evaluation and saving.

Usage:
    python train.py                    # full pipeline
    python train.py --skip-features    # skip feature engineering
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


def main(args):
    # ── 1. Load data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FRAUD DETECTION PIPELINE")
    print("=" * 60)

    loader = DataLoader(args.data_path)
    df = loader.load_data()

    # ── 2. Split ───────────────────────────────────────────
    train_df, val_df, test_df = loader.split(df)

    # ── 3. Preprocess (scale Amount & Time) ────────────────
    preprocessor = DataPreprocessor()
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # ── 4. Feature engineering ─────────────────────────────
    if not args.skip_features:
        fe = FeatureEngineer()
        train_df = fe.transform(train_df)
        val_df = fe.transform(val_df)
        test_df = fe.transform(test_df)
        print(f"Features after engineering: {train_df.shape[1] - 1}")

    # ── 5. Separate X and y ────────────────────────────────
    X_train, y_train = loader.get_feature_targets(train_df)
    X_val, y_val = loader.get_feature_targets(val_df)
    X_test, y_test = loader.get_feature_targets(test_df)

    # ── 6. Train models ────────────────────────────────────
    trainer = ModelTrainer(experiment_name=args.experiment)

    results = trainer.train(X_train, y_train, X_val, y_val)

    # ── 7. Evaluate on test set ────────────────────────────
    test_metrics = trainer.evaluate_test(X_test, y_test)

    # ── 8. Save model and preprocessor ────────────────────
    trainer.save(args.model_dir)

    # Save the fitted preprocessor so the API can apply
    # the exact same scaling at inference time.
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl")
    print(f"Preprocessor saved to {model_dir}/")

    # ── 9. Save results ────────────────────────────────────
    output = {
        "validation_results": results,
        "test_results": test_metrics,
        "best_model": trainer.best_model_name,
    }
    results_path = Path(args.model_dir) / "results.json"
    # default=str handles non-serialisable types like numpy floats
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print("Pipeline complete!")


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
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering",
    )
    args = parser.parse_args()
    main(args)