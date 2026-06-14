"""
Train and deploy the Cost-Sensitive Medical XGBoost model.

This is the production counterpart to `train_model.py` (the standard-log-loss baseline). It
trains the winning cost-sensitive objective — chosen by `compare_models.py` and recorded in
results/winner.txt, defaulting to Focal loss — on the same 80/20 split, then overwrites the
deployed model file `xgboost_diabetes_model.pkl` that `main.py` loads.

The objective is a picklable callable instance (see custom_objectives.py), so joblib can
serialize the fitted estimator and the FastAPI service can reload it. Inference never calls
the objective; it only needs the trees plus a manual sigmoid on the raw margin (main.py).

Run:  ./venv/bin/python train_cost_sensitive_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from custom_objectives import (
    sigmoid, weighted_logloss_obj, focal_loss_obj, exponential_fn_obj,
)
# Reuse the exact hyperparameters and cost settings the comparison used, so the deployed
# model matches the one the paper evaluated.
from compare_models import (
    SHARED, ZERO_AS_MISSING, COLUMNS, load_data,
    W_FN, GAMMA_FOCAL, ALPHA_FOCAL, GAMMA_EXP,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "xgboost_diabetes_model.pkl")
SEED = 42

# Map the winner name to its objective. Default to Focal (best F2/recall in our runs).
OBJECTIVES = {
    "Weighted CE": weighted_logloss_obj(W_FN),
    "Focal": focal_loss_obj(GAMMA_FOCAL, ALPHA_FOCAL),
    "Exponential": exponential_fn_obj(GAMMA_EXP),
}


def chosen_winner():
    path = os.path.join(HERE, "results", "winner.txt")
    if os.path.exists(path):
        with open(path) as f:
            name = f.read().strip()
            if name in OBJECTIVES:
                return name
    return "Focal"


def recall_fn(model, X, y):
    proba = sigmoid(model.predict(X, output_margin=True))
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), int(fn)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    winner = chosen_winner()
    print(f"Training Cost-Sensitive Medical XGBoost  (objective = {winner})...")

    # --- Baseline (standard log-loss) for an apples-to-apples printout ---
    baseline = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=SEED, n_jobs=-1, **SHARED)
    baseline.fit(X_train, y_train)

    # --- Cost-sensitive model (deployed) ---
    model = XGBClassifier(objective=OBJECTIVES[winner], random_state=SEED, n_jobs=-1, **SHARED)
    model.fit(X_train, y_train)

    b_rec, b_fn = recall_fn(baseline, X_test, y_test)
    c_rec, c_fn = recall_fn(model, X_test, y_test)
    n_pos = int((y_test == 1).sum())
    print(f"\nHeld-out test set ({len(y_test)} patients, {n_pos} diabetic):")
    print(f"  Standard log-loss   : recall {b_rec:.3f}  |  false negatives {b_fn}/{n_pos}")
    print(f"  Cost-sensitive ({winner}): recall {c_rec:.3f}  |  false negatives {c_fn}/{n_pos}")
    print(f"  -> {b_fn - c_fn} fewer diabetics missed on this split.")

    # Sanity: the deployed model must reload and predict valid probabilities.
    joblib.dump(model, MODEL_FILE)
    reloaded = joblib.load(MODEL_FILE)
    proba = sigmoid(reloaded.predict(X_test, output_margin=True))
    assert proba.min() >= 0.0 and proba.max() <= 1.0, "reloaded model gives invalid proba"

    print(f"\nDeployed cost-sensitive model -> {MODEL_FILE}")
    print("Reload check passed. main.py will now serve the safer model.")


if __name__ == "__main__":
    main()
