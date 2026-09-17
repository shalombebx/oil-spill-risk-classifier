"""
model.py
========
Model training, evaluation, cross-validation, and persistence utilities.
Refactored from 03_modeling.ipynb — Phase 5.

Usage:
    from src.model import train_xgb, evaluate, cross_validate, save_model
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


# ── Training ──────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train):
    """Majority-class Dummy Classifier — performance floor for comparison."""
    clf = DummyClassifier(strategy='most_frequent', random_state=42)
    clf.fit(X_train, y_train)
    print("Baseline (Dummy Classifier) trained")
    return clf


def train_rf(X_train, y_train, n_estimators: int = 200) -> RandomForestClassifier:
    """Train a class-balanced Random Forest classifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"Random Forest trained ({n_estimators} trees)")
    return clf


def train_xgb(X_train, y_train, n_estimators: int = 200,
              learning_rate: float = 0.1, max_depth: int = 5) -> XGBClassifier:
    """Train an XGBoost gradient boosting classifier."""
    clf = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"XGBoost trained ({n_estimators} estimators, lr={learning_rate}, depth={max_depth})")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, class_names: list, model_name: str = 'Model') -> np.ndarray:
    """
    Print classification report and return predictions.

    Returns:
        y_pred (np.ndarray): Predicted labels.
    """
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print('='*50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    return y_pred


def plot_confusion_matrix(model, X_test, y_test, class_names: list,
                          title: str = 'Confusion Matrix', save_path: str = None):
    """Plot and optionally save the confusion matrix for a fitted model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=class_names,
        cmap='Blues', ax=ax, colorbar=False
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def cross_validate(model, X, y, n_splits: int = 5) -> np.ndarray:
    """
    Run stratified k-fold cross-validation and report Weighted F1.

    Returns:
        scores (np.ndarray): F1 scores per fold.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
    print(f"CV Weighted F1 ({n_splits}-fold): {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def compare_models(models: dict, X_test, y_test, class_names: list):
    """
    Print a comparison table of Weighted F1 scores across multiple models.

    Args:
        models (dict): {'Model Name': fitted_model}
    """
    from sklearn.metrics import f1_score
    print(f"\n{'Model':<25} {'Weighted F1':>12}")
    print('-' * 40)
    for name, model in models.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name:<25} {f1:>12.4f}")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(obj, path: str):
    """Serialise any object (model, encoder, dict) to disk."""
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_model(path: str):
    """Load a serialised object from disk."""
    obj = joblib.load(path)
    print(f"Loaded: {path}")
    return obj
