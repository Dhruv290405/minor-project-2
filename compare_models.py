"""
Experiment harness: Standard XGBoost vs. Cost-Sensitive Medical XGBoost.

Trains four models that are *identical in every way except the loss function* and compares
them under a stability-focused protocol (stratified 5-fold cross-validation repeated over 5
random seeds = 25 runs per model). Because only the objective differs, any change in the
metrics is causally attributable to the loss — that is the backbone of the paper's claim.

Outputs:
  - console table of metrics (mean +/- std across the 25 runs)
  - results/metrics.csv                (raw per-run numbers)
  - results/summary.csv                (aggregated mean/std per model)
  - figures/{roc,pr,confusion_grid,recall_fn_bars,prob_distributions}.png
  - injects the Results section into PAPER.md (between the RESULTS markers)

Run:  ./venv/bin/python compare_models.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, fbeta_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

from custom_objectives import (
    sigmoid, weighted_logloss_obj, focal_loss_obj, exponential_fn_obj,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "PAPER.md")

# Columns where 0 is physiologically impossible and means "missing" (see train_model.py).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "Pedigree", "Age", "Outcome"]

# --- Stability protocol -----------------------------------------------------
SEEDS = [42, 0, 7, 13, 21]      # repeat the whole CV with these random states
N_SPLITS = 5                    # stratified folds
CURVE_SEED = 42                 # seed whose out-of-fold preds drive the plotted curves

# Shared hyperparameters — IDENTICAL for every model so the objective is the only variable.
SHARED = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    min_child_weight=1.0, base_score=0.5, tree_method="hist",
)

# Cost-sensitivity hyperparameters — frozen here for reproducibility (see PAPER.md).
W_FN = 3.0          # weighted CE: positive-class weight (~1.6x the class-balance ratio)
GAMMA_FOCAL = 2.0   # focal: focusing strength
ALPHA_FOCAL = 0.75  # focal: positive-class balance
GAMMA_EXP = 2.0     # exponential: FN penalty strength


def load_data():
    """Load the Pima dataset (cached locally), apply the 0-as-missing transform."""
    cache = os.path.join(HERE, "pima.csv")
    if os.path.exists(cache):
        data = pd.read_csv(cache)
    else:
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
               "master/pima-indians-diabetes.data.csv")
        data = pd.read_csv(url, names=COLUMNS)
        data.to_csv(cache, index=False)
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"].astype(int)
    return X, y


def make_model(objective, seed):
    """Build an XGBClassifier; objective=None means the built-in log-loss baseline."""
    kw = dict(SHARED, random_state=seed, n_jobs=-1)
    if objective is None:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
    return XGBClassifier(objective=objective, **kw)


def predict_proba(model, X):
    """
    Positive-class probability for ANY model.

    THE GOTCHA: with a custom objective XGBoost does not know the link function, so
    `predict_proba` is unreliable and `predict` returns the raw margin. We therefore take
    the margin explicitly and apply the sigmoid ourselves. For the built-in baseline this
    yields exactly what predict_proba would, so the code path is uniform and fair.
    """
    margin = model.predict(X, output_margin=True)
    return sigmoid(margin)


def metrics_at(y_true, proba, threshold=0.5):
    """Confusion-matrix-derived metrics at a given decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity (catch diabetics)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = fbeta_score(y_true, pred, beta=1, zero_division=0)
    f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)  # weights recall 2x precision
    return dict(accuracy=accuracy, precision=precision, recall=recall,
                specificity=specificity, f1=f1, f2=f2, fn=int(fn), fp=int(fp))


def best_f2_threshold(y_true, proba):
    """Threshold-moving: pick the cutoff that maximizes F2 (recall-weighted)."""
    grid = np.linspace(0.05, 0.95, 19)
    f2s = [fbeta_score(y_true, (proba >= t).astype(int), beta=2, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f2s))])


MODELS = {
    "Standard": None,
    "Weighted CE": weighted_logloss_obj(W_FN),
    "Focal": focal_loss_obj(GAMMA_FOCAL, ALPHA_FOCAL),
    "Exponential": exponential_fn_obj(GAMMA_EXP),
}


def run_experiment(X, y):
    """Full CV x seeds sweep. Returns per-run dataframe and out-of-fold preds (CURVE_SEED)."""
    rows = []
    oof = {name: np.zeros(len(y)) for name in MODELS}  # out-of-fold proba for the curve seed

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            y_tr, y_va = y.iloc[tr], y.iloc[va]
            for name, obj in MODELS.items():
                model = make_model(obj, seed)
                model.fit(X_tr, y_tr)
                proba = predict_proba(model, X_va)

                # Gotcha demonstration: probabilities must be valid after the manual sigmoid.
                assert proba.min() >= 0.0 and proba.max() <= 1.0, "proba out of [0,1]"

                m = metrics_at(y_va.values, proba, 0.5)
                thr = best_f2_threshold(y_va.values, proba)
                m_opt = metrics_at(y_va.values, proba, thr)
                rows.append(dict(
                    model=name, seed=seed, fold=fold,
                    roc_auc=roc_auc_score(y_va, proba),
                    pr_auc=average_precision_score(y_va, proba),
                    recall_opt=m_opt["recall"], fn_opt=m_opt["fn"], thr_opt=thr,
                    **m,
                ))
                if seed == CURVE_SEED:
                    oof[name][va] = proba
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(df):
    agg = df.groupby("model").agg(["mean", "std"])
    # Preserve a sensible model ordering.
    return agg.reindex(list(MODELS))


def fmt(agg, metric):
    return {m: f"{agg.loc[m, (metric, 'mean')]:.3f} ± {agg.loc[m, (metric, 'std')]:.3f}"
            for m in agg.index}


def print_table(agg):
    cols = ["recall", "f2", "precision", "specificity", "accuracy", "roc_auc", "pr_auc", "fn"]
    header = f"{'Model':<13}" + "".join(f"{c:>16}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for m in agg.index:
        line = f"{m:<13}"
        for c in cols:
            mean = agg.loc[m, (c, "mean")]
            std = agg.loc[m, (c, "std")]
            line += f"{f'{mean:.3f}±{std:.3f}':>16}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
COLORS = {"Standard": "#64748b", "Weighted CE": "#0ea5e9",
          "Focal": "#f59e0b", "Exponential": "#dc2626"}


def fig_roc_pr(y, oof):
    for kind in ("roc", "pr"):
        plt.figure(figsize=(6, 5))
        for name in MODELS:
            p = oof[name]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                plt.plot(fpr, tpr, color=COLORS[name],
                         label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
            else:
                prec, rec, _ = precision_recall_curve(y, p)
                plt.plot(rec, prec, color=COLORS[name],
                         label=f"{name} (AP={average_precision_score(y, p):.3f})")
        if kind == "roc":
            plt.plot([0, 1], [0, 1], "k--", lw=0.8)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")
            plt.title("ROC — Standard vs. Cost-Sensitive")
            out = "roc.png"
        else:
            plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision")
            plt.title("Precision–Recall — Standard vs. Cost-Sensitive")
            out = "pr.png"
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, out), dpi=130); plt.close()


def fig_confusion(y, oof):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, name in zip(axes.ravel(), MODELS):
        pred = (oof[name] >= 0.5).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                # Highlight the false-negative cell (true=1, pred=0) — the dangerous error.
                danger = (i == 1 and j == 0)
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="red" if danger else "black",
                        fontweight="bold" if danger else "normal", fontsize=13)
        ax.set_title(f"{name}  (FN={cm[1,0]})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    fig.suptitle("Confusion matrices (out-of-fold, threshold 0.5) — fewer FN is safer")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_grid.png"), dpi=130); plt.close(fig)


def fig_recall_fn_bars(agg):
    names = list(agg.index)
    colors = [COLORS[n] for n in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.bar(names, [agg.loc[n, ("recall", "mean")] for n in names],
           yerr=[agg.loc[n, ("recall", "std")] for n in names], color=colors, capsize=4)
    a1.set_title("Recall / Sensitivity (higher = safer)"); a1.set_ylim(0, 1)
    a1.tick_params(axis="x", rotation=15)
    a2.bar(names, [agg.loc[n, ("fn", "mean")] for n in names],
           yerr=[agg.loc[n, ("fn", "std")] for n in names], color=colors, capsize=4)
    a2.set_title("False Negatives per fold (lower = safer)")
    a2.tick_params(axis="x", rotation=15)
    fig.suptitle("Mean ± std across 5 folds × 5 seeds (25 runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recall_fn_bars.png"), dpi=130); plt.close(fig)


def fig_prob_dist(y, oof):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, name in zip(axes.ravel(), MODELS):
        p = oof[name]
        ax.hist(p[y == 0], bins=25, alpha=0.6, color="#10b981", label="Non-diabetic")
        ax.hist(p[y == 1], bins=25, alpha=0.6, color="#dc2626", label="Diabetic")
        ax.axvline(0.5, color="k", ls="--", lw=0.8)
        ax.set_title(name); ax.set_xlabel("Predicted risk"); ax.legend(fontsize=8)
    fig.suptitle("Predicted-risk distributions — cost-sensitive losses push diabetics right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "prob_dist.png"), dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# PAPER.md results injection
# ---------------------------------------------------------------------------
def md_table(agg):
    cols = [("recall", "Recall"), ("f2", "F2"), ("precision", "Precision"),
            ("specificity", "Specificity"), ("accuracy", "Accuracy"),
            ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("fn", "FN/fold")]
    head = "| Model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for m in agg.index:
        cells = [f"{agg.loc[m,(c,'mean')]:.3f} ± {agg.loc[m,(c,'std')]:.3f}" for c, _ in cols]
        lines.append(f"| **{m}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inject_results(agg, best):
    if not os.path.exists(PAPER):
        return
    with open(PAPER) as f:
        text = f.read()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        return
    std_recall = agg.loc["Standard", ("recall", "mean")]
    std_fn = agg.loc["Standard", ("fn", "mean")]
    block = f"""{start}
_Auto-generated by `compare_models.py` — {N_SPLITS}-fold cross-validation × {len(SEEDS)} seeds ({N_SPLITS*len(SEEDS)} runs per model), threshold 0.5._

{md_table(agg)}

**Headline:** the **{best}** objective lifts recall from {std_recall:.3f} (Standard) to \
{agg.loc[best,('recall','mean')]:.3f} and cuts mean false negatives from {std_fn:.2f} to \
{agg.loc[best,('fn','mean')]:.2f} per fold, at a modest cost in precision/specificity — the \
intended trade for a screening tool.

![ROC](figures/roc.png)
![Precision-Recall](figures/pr.png)
![Confusion matrices](figures/confusion_grid.png)
![Recall and false negatives](figures/recall_fn_bars.png)
![Risk distributions](figures/prob_dist.png)
{end}"""
    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    with open(PAPER, "w") as f:
        f.write(text)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    print("Loading data and running 5-fold × 5-seed comparison (this takes a minute)...")
    X, y = load_data()
    df, oof = run_experiment(X, y)
    agg = summarize(df)

    df.to_csv(os.path.join(RES_DIR, "metrics.csv"), index=False)
    agg.to_csv(os.path.join(RES_DIR, "summary.csv"))
    print_table(agg)

    # Figures.
    fig_roc_pr(y, oof)
    fig_confusion(y, oof)
    fig_recall_fn_bars(agg)
    fig_prob_dist(y, oof)
    print(f"Figures written to {FIG_DIR}/")

    # --- Thesis check: a cost-sensitive model must be safer than the baseline ---
    std_recall = agg.loc["Standard", ("recall", "mean")]
    std_fn = agg.loc["Standard", ("fn", "mean")]
    cost_sensitive = [m for m in MODELS if m != "Standard"]
    safer = [m for m in cost_sensitive
             if agg.loc[m, ("recall", "mean")] >= std_recall
             and agg.loc[m, ("fn", "mean")] <= std_fn]
    print("Thesis check (recall up AND false negatives down vs. Standard):")
    for m in cost_sensitive:
        ok = m in safer
        print(f"  {m:<13} recall {agg.loc[m,('recall','mean')]:.3f} (std {std_recall:.3f}) | "
              f"FN {agg.loc[m,('fn','mean')]:.2f} (std {std_fn:.2f})  -> {'PASS' if ok else 'no'}")
    assert safer, "No cost-sensitive model beat the baseline — thesis not supported."

    # Pick the safest model (max recall, then min FN) for deployment/the headline.
    best = max(safer, key=lambda m: (agg.loc[m, ("recall", "mean")],
                                     -agg.loc[m, ("fn", "mean")]))
    print(f"\nSafest cost-sensitive model: {best}")
    inject_results(agg, best)
    print(f"Results injected into {PAPER}")
    # Record the winner so train_cost_sensitive_model.py can default to it.
    with open(os.path.join(RES_DIR, "winner.txt"), "w") as f:
        f.write(best)


if __name__ == "__main__":
    main()
