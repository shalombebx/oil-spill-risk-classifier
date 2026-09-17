"""
explain.py
==========
SHAP explainability wrappers and plot helpers.
Refactored from 04_explainability.ipynb — Phase 6.

Usage:
    from src.explain import get_shap_values, plot_global, plot_beeswarm, plot_force
"""

import shap
import numpy as np
import matplotlib.pyplot as plt


# ── SHAP computation ──────────────────────────────────────────────────────────

def get_shap_values(model, X_test) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    Returns:
        explainer   (shap.TreeExplainer): Fitted explainer object.
        shap_values (list of np.ndarray): One array per class for multiclass.
    """
    print("Computing SHAP values... (may take 1–2 minutes)")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("Done")
    return explainer, shap_values


# ── Global explanations ───────────────────────────────────────────────────────

def plot_global(shap_values, X_test, feature_names: list, class_names: list,
                save_path: str = None):
    """
    Global feature importance bar chart — aggregated across all classes.
    Answers: 'Which features matter most across the entire dataset?'
    """
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        class_names=class_names,
        plot_type='bar',
        show=False
    )
    plt.title('SHAP — Global Feature Importance by Severity Class')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_beeswarm(shap_values, X_test, feature_names: list,
                  class_idx: int, class_name: str, save_path: str = None):
    """
    Beeswarm plot for one class — shows direction and magnitude of feature influence.
    Answers: 'How does each feature push the model toward or away from this class?'

    Args:
        class_idx: Index of the target class (check le_target.classes_ for order).
    """
    shap.summary_plot(
        shap_values[class_idx], X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Detail — Class: {class_name}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


# ── Local explanations ────────────────────────────────────────────────────────

def plot_force(explainer, shap_values, X_test, idx: int,
               class_idx: int, feature_names: list,
               true_label: str, pred_label: str, save_path: str = None):
    """
    SHAP force plot for a single prediction.
    Answers: 'Why did the model predict this specific incident as Major/Minor?'

    Args:
        idx: Row index in X_test to explain.
        class_idx: Which class's SHAP values to use.
    """
    print(f"Sample {idx}  |  True: {true_label}  |  Predicted: {pred_label}")

    shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][idx],
        X_test.iloc[idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        figsize=(16, 3)
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


# ── Interpretation helpers ────────────────────────────────────────────────────

def top_features(shap_values, feature_names: list, class_idx: int,
                 class_name: str, top_n: int = 5):
    """
    Print the top N features driving predictions for a given class,
    ranked by mean absolute SHAP value.
    """
    mean_abs = np.abs(shap_values[class_idx]).mean(axis=0)
    ranked   = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} features for class '{class_name}':")
    print(f"{'Feature':<25} {'Mean |SHAP|':>12}")
    print('-' * 40)
    for name, score in ranked[:top_n]:
        print(f"{name:<25} {score:>12.4f}")
