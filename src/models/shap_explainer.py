"""
SHAP-based model interpretability for the fraud detection pipeline.

Called after the best model is identified.  Produces two plots and prints
the top-5 features by mean absolute SHAP value to stdout.

Supported model families:
    - Tree-based (Random Forest, XGBoost, LightGBM) → shap.TreeExplainer
    - Linear (Logistic Regression)                  → shap.LinearExplainer
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ── Constants ─────────────────────────────────────────────────────────────────

_TREE_MODEL_NAMES = {"RandomForest", "XGBoost", "LightGBM"}
_LINEAR_MODEL_NAMES = {"LogisticRegression"}


# ── Public entry point ────────────────────────────────────────────────────────

def explain(
    model,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs/shap",
) -> None:
    """
    Compute SHAP values for the winning model and save interpretability plots.

    Parameters
    ----------
    model       : fitted estimator (sklearn, XGBoost, or LightGBM)
    model_name  : one of 'LogisticRegression', 'RandomForest',
                  'XGBoost', 'LightGBM'
    X_test      : feature matrix (test split)
    y_test      : true labels (test split)
    output_dir  : directory where PNG files are written
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print(f"  SHAP Interpretability  [{model_name}]")
    print(f"{'-' * 60}")

    # ── 1. Choose the right explainer ────────────────────────────────────────
    explainer, shap_values = _build_explainer(model, model_name, X_test)

    # shap_values may be a list (one array per class) for multi-output models;
    # for binary classifiers we always want the positive-class (fraud) values.
    sv = _extract_positive_class(shap_values)

    # ── 2. Summary plot — global feature importance ──────────────────────────
    _save_summary_plot(sv, X_test, out)

    # ── 3. Waterfall plot — single highest-risk prediction ───────────────────
    _save_waterfall_plot(model, explainer, shap_values, sv, X_test, y_test, out)

    # ── 4. Top-5 features by mean |SHAP value| ──────────────────────────────
    _print_top5(sv, X_test)

    print(f"\n  Plots saved to {out}/")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_explainer(model, model_name: str, X_test: pd.DataFrame):
    """Instantiate the correct SHAP explainer and compute SHAP values."""
    if model_name in _TREE_MODEL_NAMES:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_name in _LINEAR_MODEL_NAMES:
        # LinearExplainer needs a background dataset (the test set itself
        # is fine for a quick explanation pass)
        explainer = shap.LinearExplainer(
            model,
            X_test,
            feature_perturbation="correlation_dependent",
        )
        shap_values = explainer.shap_values(X_test)
    else:
        # Fallback: KernelExplainer on a small sample to keep runtime sane
        print(
            f"  Warning: '{model_name}' is not a known tree/linear model; "
            "falling back to KernelExplainer on 200 samples."
        )
        background = shap.sample(X_test, 200)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)

    return explainer, shap_values


def _extract_positive_class(shap_values) -> np.ndarray:
    """
    Return the SHAP value matrix for the fraud class.

    sklearn tree-based models (RandomForest) return a list [neg_sv, pos_sv];
    XGBoost and LightGBM return a single 2-D array directly.
    """
    if isinstance(shap_values, list):
        # list of arrays — index 1 is the positive (fraud) class
        return shap_values[1]
    # single array — already the positive class
    return shap_values


def _save_summary_plot(
    sv: np.ndarray,
    X_test: pd.DataFrame,
    out: Path,
) -> None:
    """Generate and save the global SHAP summary (beeswarm) plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, show=False, plot_size=None)
    plt.title("SHAP Feature Importance — Global (Test Set)", fontsize=13, pad=12)
    plt.tight_layout()
    path = out / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [+] shap_summary.png saved  ({path})")


def _save_waterfall_plot(
    model,
    explainer,
    shap_values,          # raw output from explainer (may be list)
    sv: np.ndarray,       # positive-class 2-D array
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out: Path,
) -> None:
    """
    Generate and save a waterfall plot for the single highest-risk
    transaction (the sample with the highest predicted fraud probability).
    """
    # Find the highest predicted fraud probability in the test set
    proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(proba))
    high_risk_prob = float(proba[high_risk_idx])
    true_label = int(y_test.iloc[high_risk_idx])

    print(
        f"  High-risk sample: index {high_risk_idx}, "
        f"P(fraud)={high_risk_prob:.4f}, true label={true_label}"
    )

    # Build an Explanation object for the waterfall plot
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    shap_row = sv[high_risk_idx]

    explanation = shap.Explanation(
        values=shap_row,
        base_values=float(base_value),
        data=X_test.iloc[high_risk_idx].values,
        feature_names=list(X_test.columns),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title(
        f"SHAP Waterfall — Highest-Risk Transaction "
        f"(P(fraud)={high_risk_prob:.4f})",
        fontsize=12,
        pad=12,
    )
    plt.tight_layout()
    path = out / "shap_waterfall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [+] shap_waterfall.png saved  ({path})")


def _print_top5(sv: np.ndarray, X_test: pd.DataFrame) -> None:
    """Compute and print the top-5 features by mean |SHAP value|."""
    mean_abs = np.abs(sv).mean(axis=0)
    feature_names = list(X_test.columns)
    ranked = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n  +-----------------------------------------------------+")
    print("  |  Top 5 Features by Mean |SHAP Value|               |")
    print("  +---+----------------------------+--------------------+")
    print("  | # | Feature                    | Mean |SHAP|        |")
    print("  +---+----------------------------+--------------------+")
    for rank, (feat, score) in enumerate(ranked[:5], start=1):
        print(f"  | {rank} | {feat:<26} | {score:>18.6f} |")
    print("  +-----------------------------------------------------+")
