from tqdm import tqdm
import glob
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_validate, GridSearchCV
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import sklearn
import cupy as cp
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from cuml.svm import SVC
import pickle
import random
import re

print(sklearn.__version__)
print(cuml.__version__)

os.environ["CUPY_GPU_MEMORY_LIMIT"] = "90%"
#from rmm.allocators.cupy import rmm_cupy_allocator

# Route CuPy allocations through RAPIDS RMM
#cp.cuda.set_allocator(rmm_cupy_allocator)

# ---------------------------------------------
# Reproducibility
# ---------------------------------------------
RANDOM_SEED = 42

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

set_all_seeds(RANDOM_SEED)

# ---------------------------------------------
# Metrics helper
# ---------------------------------------------
def report_np(y_true, y_pred, n_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.arange(n_classes)[None, :]
    supp = classes == y_true[:, None]
    tmp  = classes == y_pred[:, None]
    hits = (tmp & supp).sum(axis=0)
    pred = tmp.sum(axis=0)
    n    = y_true.shape[0]

    supp     = supp.sum(axis=0)
    pred_inv = np.array([1/i if i != 0 else 0 for i in pred])
    prec     = hits * pred_inv
    supp_inv = np.array([1/i if i != 0 else 0 for i in supp])
    rec      = hits * supp_inv

    balanced_acc = rec.mean()
    prec_rec     = prec + rec
    prec_rec_mult = 2 * prec * rec
    prec_rec_inv  = np.array([1/i if i != 0 else 0 for i in prec_rec])
    f1            = prec_rec_mult * prec_rec_inv

    acc      = hits.sum() / n
    stacked  = np.vstack([prec, rec, f1])
    macro    = stacked.mean(axis=1)
    weighted = stacked @ supp / n

    return hits, pred - hits, acc, balanced_acc, supp, prec, rec, f1, macro, weighted


def cr(y, x, num_labels, digits=4):
    hits, nonhits, acc, balanced_acc, supp, prec, rec, f1, macro, weighted = \
        report_np(y, x, num_labels)
    return round(acc, digits), round(macro[2], digits)

# ---------------------------------------------
# Label mappings
# ---------------------------------------------
SAFETY_ORDER = {
    '__casual__':                 4,
    '__possibly_needs_caution__': 3,
    '__probably_needs_caution__': 2,
    '__needs_caution__':          1,
    '__needs_intervention__':     0,
}
REVERSE_SAFETY_ORDER = {v: k for k, v in SAFETY_ORDER.items()}

BINARY_SAFETY_ORDER = {
    '__casual__':                 1,
    '__possibly_needs_caution__': 0,
    '__probably_needs_caution__': 0,
    '__needs_caution__':          0,
    '__needs_intervention__':     0,
}
TERNARY_SAFETY_ORDER = {
    '__casual__':                 2,
    '__possibly_needs_caution__': 1,
    '__probably_needs_caution__': 1,
    '__needs_caution__':          1,
    '__needs_intervention__':     0,
}
SAFETY_TIER_LABELS = {
    'high_safety': ['__casual__', '__possibly_needs_caution__'],
    'low_safety':  ['__needs_caution__', '__needs_intervention__'],
}

# ---------------------------------------------
# Feature column lists
# ---------------------------------------------
LDA    = ['predict_conseq', 'chance_responsblt', 'urgency', 'social_norms',
          'predict_event', 'chance_control', 'pleasantness', 'goal_support',
          'other_control']
Ranked = ['social_norms', 'other_responsblt', 'standards', 'other_control',
          'self_responsblt', 'self_control', 'goal_support', 'goal_relevance',
          'suddenness', 'unpleasantness', 'pleasantness', 'not_consider',
          'predict_conseq', 'predict_event']

# ---------------------------------------------
# Load data
# ---------------------------------------------
appraisals_diasafety = pd.read_csv('prosocial-appraised/diasafety_all_appraised.csv')
gemma_diasafety = pd.read_csv('diasafety_gemma-2-2b_scores.csv.gz')
appraisals_diasafety['gemma-2-2b_label'] = gemma_diasafety['label']

context_appraisals  = [x for x in appraisals_diasafety.columns if 'prompt_'   in x]
response_appraisals = [x for x in appraisals_diasafety.columns if 'response_' in x]
delta_appraisals    = [x for x in appraisals_diasafety.columns if 'delta_'    in x]

# ---------------------------------------------
# Column groups
# ---------------------------------------------
GEMMA_binary_cols  = ['gemma-2-2b_label']
LDA_cols           = [f'prompt_{x}'   for x in LDA]    + [f'response_{x}' for x in LDA]
Ranked_cols        = [f'prompt_{x}'   for x in Ranked] + [f'response_{x}' for x in Ranked]
appraisals_cols    = context_appraisals + response_appraisals

# ---------------------------------------------
# Deduplication / filtering
# ---------------------------------------------
appraisals_diasafety.drop_duplicates(subset=['prompt','response'], inplace=True)

# ---------------------------------------------
# Classifiers
# ---------------------------------------------
classifiers = {
    "LR":  LogisticRegression(max_iter=1000),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "SVM": SVC(kernel='rbf'),
}

# ---------------------------------------------
# Hyperparameter grids
# ---------------------------------------------
param_grids = {
    "LR": {
        "clf__C":       [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2", None],
    },
    "RF": {
        "clf__n_estimators":      [50, 100, 200],
        "clf__max_depth":         [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    "SVM": {
        "clf__C":     [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto"],
    },
}

# ---------------------------------------------
# CV setup
# ---------------------------------------------
CV_SPLITS = 10
cv      = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]

# ---------------------------------------------
# Feature sets
# ---------------------------------------------
model_versions = {
    "All appraisals":                   appraisals_cols,
    "Ranked appraisals":                Ranked_cols,
    "LDA appraisals":                   LDA_cols,
    "Gemma":                      GEMMA_binary_cols,
    "Gemma + All appraisals":     GEMMA_binary_cols + appraisals_cols,
    "Gemma + Ranked appraisals":  GEMMA_binary_cols + Ranked_cols,
    "Gemma + LDA appraisals":     GEMMA_binary_cols + LDA_cols,
}

diasafety_train = appraisals_diasafety[appraisals_diasafety.split == "train"].drop_duplicates(subset=['prompt','response']).reset_index()
diasafety_valid = appraisals_diasafety[appraisals_diasafety.split == "val"].reset_index()
diasafety_test = appraisals_diasafety[appraisals_diasafety.split == "test"].reset_index()

# ---------------------------------------------
# PHASE 1 Hyperparameter tuning on the first feature set
# ---------------------------------------------
first_feat_name, first_cols = next(iter(model_versions.items()))
X_tune = diasafety_train[first_cols].values
y_tune = np.array([1 if x == "Safe" else 0 for x in diasafety_train["label"]])

print(f"\n{'='*60}")
print(f"Hyperparameter tuning on feature set: '{first_feat_name}'")
print(f"{'='*60}")

best_global_score = -np.inf
best_clf_name     = None
best_pipeline     = None
tuning_results    = []

for clf_name, clf in classifiers.items():
    print(f"\n  Tuning {clf_name}...")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])

    search = GridSearchCV(
        pipe,
        param_grids[clf_name],
        cv=cv,
        scoring="f1_macro",
        refit=True,          # refit best params on full X_tune
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_tune, y_tune)

    # Mean + std of macro F1 across CV folds for the best parameter combo
    best_idx  = search.best_index_
    mean_f1   = search.cv_results_["mean_test_score"][best_idx]
    std_f1    = search.cv_results_["std_test_score"][best_idx]

    tuning_results.append({
        "Classifier":       clf_name,
        "Features":         first_feat_name,
        "Best params":      search.best_params_,
        "CV macro F1 mean": round(mean_f1, 4),
        "CV macro F1 std":  round(std_f1, 4),
    })

    print(f"  {clf_name}: macro F1 = {mean_f1:.4f} + {std_f1:.4f}  |  {search.best_params_}")

    if mean_f1 > best_global_score:
        best_global_score = mean_f1
        best_clf_name     = clf_name
        best_pipeline     = search.best_estimator_   # already refit on full X_tune
        best_search       = search                   # keep for potential inspection

print(f"\n{'='*60}")
print(f"Best classifier : {best_clf_name}")
print(f"CV macro F1     : {best_global_score:.4f}")
print(f"Best params     : {best_search.best_params_}")
print(f"{'='*60}\n")

# ---------------------------------------------
# PHASE 2 Full CV evaluation (all feature sets, best clf only)
# ---------------------------------------------

def evaluate_classifier(clf, X, y, cv):
    """Returns mean + std for each metric across CV folds."""
    pipe   = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=METRICS,
                            return_train_score=True, n_jobs=-1)
    return {
        metric: {
            "mean": scores[f"test_{metric}"].mean(),
            "std":  scores[f"test_{metric}"].std(),
            "folds": scores[f"test_{metric}"]
        }
        for metric in METRICS
    }

# Extract the tuned estimator step so we can clone it for other feature sets.
# best_pipeline is a fitted Pipeline; grab the clf step's parameters.
best_params_clean = {
    k.replace("clf__", ""): v
    for k, v in best_search.best_params_.items()
}
best_clf_tuned = clone(classifiers[best_clf_name]).set_params(**best_params_clean)

print(f"Running full CV + hold-out evaluation with tuned {best_clf_name} "
      f"across all feature sets...\n")

results = []

for feat_name, cols in tqdm(model_versions.items(), desc="Feature sets"):
    X = diasafety_train[cols].values
    y = np.array([1 if x == "Safe" else 0 for x in diasafety_train["label"]])

    # Cross-validation on train set
    cv_scores = evaluate_classifier(best_clf_tuned, X, y, cv)

    # Refit on full train set, then evaluate on hold-out sets
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(best_clf_tuned))])
    pipe.fit(X, y)

    for test_name, test_df in [("valid",      diasafety_valid),
                                ("test", diasafety_test)]:
        X_test = test_df[cols].values
        y_test = np.array([1 if x == "Safe" else 0 for x in test_df["label"]])
        preds  = pipe.predict(X_test)

        row = {
            "Classifier": best_clf_name,
            "Features":   feat_name,
            "Test set":   test_name,
            "Accuracy":   round((preds == y_test).mean(), 4),
            "macro F1":   round(f1_score(y_test, preds, average="macro"), 4),
            "seed":       RANDOM_SEED,
            "cv_folds":   CV_SPLITS,
        }

        # Attach CV mean + std for every metric, including macro F1
        for metric, vals in cv_scores.items():
            row[f"cv_{metric}_mean"] = round(vals["mean"], 4)
            row[f"cv_{metric}_std"]  = round(vals["std"],  4)
            row[f"cv_{metric}_folds"]  = vals["folds"]

        results.append(row)
        pd.DataFrame({"label": y_test, "preds": preds}).to_csv(f"diasafety_{test_name}_{best_clf_name}_{feat_name.replace(' ','').replace('+','')}.csv.gz", index=False)

# ---------------------------------------------
# Save outputs
# ---------------------------------------------
results_df = pd.DataFrame(results)
tuning_df  = pd.DataFrame(tuning_results)

results_df.to_csv("diasafety_results.csv.gz", index=False)
tuning_df.to_csv("diasafety_tuning_summary.csv",     index=False)

print("\nTuning summary:")
print(tuning_df.to_string(index=False))
print("\nHold-out results (macro F1 + CV macro F1 mean + std):")
display_cols = ["Classifier", "Features", "Test set", "macro F1",
                "cv_f1_macro_mean", "cv_f1_macro_std", "Accuracy"]
print(results_df[display_cols].to_string(index=False))
print("\nDone. Results saved to diasafety_results.csv.gz and diasafety_tuning_summary.csv")
