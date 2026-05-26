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
finetuned_train = pd.DataFrame()
files = glob.glob("lm-classifier_results_train/*.csv.gz")
for f in tqdm(files, total= len(files)):
   detector = f.split('/')[-1].split('\\')[-1].split('_scores')[0]
   print(detector)
   df = pd.read_csv(f)
   finetuned_train[f"{detector}_label"] = df['label']
   for i in range(0, 5):
      finetuned_train[f"{detector}_{i}_probs"] = df[f'{i}_probs']
   if ('deberta-v3-large_scores_train' == detector):
     finetuned_train[f"deberta_label"] = df['label']
     for i in range(0, 5):
      finetuned_train[f"deberta_{i}_probs"] = df[f'{i}_probs']
      
finetuned_valid = pd.DataFrame()
files = glob.glob("lm-classifier_results_valid/*.csv.gz")
for f in tqdm(files, total= len(files)):
   detector = f.split('/')[-1].split('\\')[-1].split('_scores')[0]
   print(detector)
   df = pd.read_csv(f)
   finetuned_valid[f"{detector}_label"] = df['label']
   for i in range(0, 5):
      finetuned_valid[f"{detector}_{i}_probs"] = df[f'{i}_probs']
   if ('deberta-v3-large_scores_valid' == detector):
     finetuned_valid[f"deberta_label"] = df['label']
     for i in range(0, 5):
      finetuned_valid[f"deberta_{i}_probs"] = df[f'{i}_probs']

finetuned = pd.DataFrame()
files = glob.glob("lm-classifier_results_test/*.csv.gz")
for f in tqdm(files, total= len(files)):
   detector = f.split('/')[-1].split('\\')[-1].split('_scores')[0]
   print(detector)
   df = pd.read_csv(f)
   finetuned[f"{detector}_label"] = df['label']
   for i in range(0, 5):
      finetuned[f"{detector}_{i}_probs"] = df[f'{i}_probs']
   if ('deberta-v3-large_scores' == detector):
     finetuned[f"deberta_label"] = df['label']
     for i in range(0, 5):
      finetuned[f"deberta_{i}_probs"] = df[f'{i}_probs']

prosocial_train = pd.read_csv('prosocial-appraised/prosocial_dialog_train_appraised.csv')
prosocial_valid = pd.read_csv('prosocial-appraised/prosocial_dialog_validation_appraised.csv')
prosocial_test = pd.read_csv('prosocial-appraised/prosocial_dialog_test_appraised.csv')

canary_prosocial_train = pd.read_csv('prosocial-appraised/prosocial_dialog_train_appraised_canary.csv')
canary_prosocial_valid = pd.read_csv('prosocial-appraised/prosocial_dialog_validation_appraised_canary.csv')
canary_prosocial_test = pd.read_csv('prosocial-appraised/prosocial_dialog_test_appraised_canary.csv')

for temp in [canary_prosocial_train, canary_prosocial_valid, canary_prosocial_test]:
    temp['canary_safetylabel'] = [f"__{x.split('__')[1].replace('posh_','')}__".replace('_\']','') if "__" in x else "____" for x in temp['canary']]
    temp.loc[~temp['canary_safetylabel'].isin(SAFETY_ORDER.keys()), 'canary_safetylabel'] = "unknown"
    temp['canary_safetylabelid'] = temp['canary_safetylabel'].map(SAFETY_ORDER).fillna(-1)
    
#integration of lm-finetued classifier results + canary
prosocial_train = pd.concat([prosocial_train, finetuned_train, canary_prosocial_train[['canary', 'canary_safetylabel', 'canary_safetylabelid']]], axis=1)
prosocial_valid = pd.concat([prosocial_valid, finetuned_valid, canary_prosocial_valid[['canary', 'canary_safetylabel', 'canary_safetylabelid']]], axis=1)
prosocial_test = pd.concat([prosocial_test, finetuned, canary_prosocial_test[['canary', 'canary_safetylabel', 'canary_safetylabelid']]], axis=1)

context_appraisals  = [x for x in prosocial_train.columns if 'context_'   in x and '_context' not in x]
response_appraisals = [x for x in prosocial_train.columns if 'response_' in x and '_id' not in x]
delta_appraisals = []
for x in response_appraisals: delta_appraisals.append(x.replace('response_', 'delta_'))
appraisals = context_appraisals + response_appraisals

prosocial_train[delta_appraisals] = prosocial_train[response_appraisals].values - prosocial_train[context_appraisals].values
prosocial_valid[delta_appraisals] = prosocial_valid[response_appraisals].values - prosocial_valid[context_appraisals].values
prosocial_test[delta_appraisals] = prosocial_test[response_appraisals].values - prosocial_test[context_appraisals].values

# ---------------------------------------------
# Column groups
# ---------------------------------------------
prosocial_train['gemma-2-2b_context_label_binary'] = prosocial_train['gemma-2-2b_context_label'].map(REVERSE_SAFETY_ORDER).map(BINARY_SAFETY_ORDER)
prosocial_train['canary_safetylabelid_binary'] = prosocial_train['canary_safetylabel'].map(BINARY_SAFETY_ORDER).fillna(-1)
prosocial_valid['gemma-2-2b_context_label_binary'] = prosocial_valid['gemma-2-2b_context_label'].map(REVERSE_SAFETY_ORDER).map(BINARY_SAFETY_ORDER)
prosocial_valid['canary_safetylabelid_binary'] = prosocial_valid['canary_safetylabel'].map(BINARY_SAFETY_ORDER).fillna(-1)
prosocial_test['gemma-2-2b_context_label_binary'] = prosocial_test['gemma-2-2b_context_label'].map(REVERSE_SAFETY_ORDER).map(BINARY_SAFETY_ORDER)
prosocial_test['canary_safetylabelid_binary'] = prosocial_test['canary_safetylabel'].map(BINARY_SAFETY_ORDER).fillna(-1)

GEMMA_cols = ['gemma-2-2b_context_label']
GEMMA_binary_cols = ['gemma-2-2b_context_label_binary']
CANARY_cols = ['canary_safetylabelid']
CANARY_binary_cols = ['canary_safetylabelid_binary']
LDA_cols = [f'context_{x}' for x in LDA]
Ranked_cols = [f'context_{x}' for x in Ranked]
context_appraisals = context_appraisals

# ---------------------------------------------
# Filtering
# ---------------------------------------------
prosocial_test_filtered = prosocial_test[~prosocial_test['context'].isin(prosocial_train['context']) & ~prosocial_test['context'].isin(prosocial_valid['context'])].reset_index()

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

# ---------------------------------------------
# Feature sets
# ---------------------------------------------
model_versions = {
    "Context appraisals": context_appraisals,
    "Ranked appraisals": Ranked_cols,
    "LDA appraisals": LDA_cols,
    "Canary": CANARY_cols,
    "Canary + Context appraisals": CANARY_cols + context_appraisals,
    "Canary + Ranked appraisals": CANARY_cols + Ranked_cols,
    "Canary + LDA appraisals": CANARY_cols + LDA_cols,
    "Gemma": GEMMA_cols,
    "Gemma + Context appraisals": GEMMA_cols + context_appraisals,
    "Gemma + Ranked appraisals": GEMMA_cols + Ranked_cols,
    "Gemma + LDA appraisals": GEMMA_cols + LDA_cols,
}

##################### multiclass
METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"]

# ---------------------------------------------
# PHASE 1 Hyperparameter tuning on the first feature set
# ---------------------------------------------
first_feat_name, first_cols = next(iter(model_versions.items()))
X_tune = prosocial_train[first_cols].values
y_tune = prosocial_train.safety_label.map(SAFETY_ORDER).values

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
    X = prosocial_train[cols].values
    y = prosocial_train.safety_label.map(SAFETY_ORDER).values

    # Cross-validation on train set
    cv_scores = evaluate_classifier(best_clf_tuned, X, y, cv)

    # Refit on full train set, then evaluate on hold-out sets
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(best_clf_tuned))])
    pipe.fit(X, y)

    for test_name, test_df in [("valid",      prosocial_valid),
                                ("test", prosocial_test),
                                ("filtered", prosocial_test_filtered)]:
        X_test = test_df[cols].values
        y_test = test_df.safety_label.map(SAFETY_ORDER).values
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
        pd.DataFrame({"label": y_test, "preds": preds}).to_csv(f"prosocialdialog_{test_name}_{best_clf_name}_{feat_name.replace(' ','').replace('+','')}_multiclass.csv.gz", index=False)

# ---------------------------------------------
# Save outputs
# ---------------------------------------------
results_df = pd.DataFrame(results)
tuning_df  = pd.DataFrame(tuning_results)

results_df.to_csv("prosocialdialog_results_multiclass_tuned.csv.gz", index=False)
tuning_df.to_csv("prosocialdialog_tuning_summary.csv",     index=False)

print("\nTuning summary:")
print(tuning_df.to_string(index=False))
print("\nHold-out results (macro F1 + CV macro F1 mean + std):")
display_cols = ["Classifier", "Features", "Test set", "macro F1",
                "cv_f1_macro_mean", "cv_f1_macro_std", "Accuracy"]
print(results_df[display_cols].to_string(index=False))
print("\nDone. Results saved to prosocialdialog_results_multiclass_tuned.csv.gz and prosocialdialog_tuning_summary.csv")

################################ binary
METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]

model_versions = {
    "Context appraisals": context_appraisals,
    "Ranked appraisals": Ranked_cols,
    "LDA appraisals": LDA_cols,
    "Canary": CANARY_binary_cols,
    "Canary + Context appraisals": CANARY_binary_cols + context_appraisals,
    "Canary + Ranked appraisals": CANARY_binary_cols + Ranked_cols,
    "Canary + LDA appraisals": CANARY_binary_cols + LDA_cols,
    "Gemma": GEMMA_binary_cols,
    "Gemma + Context appraisals": GEMMA_binary_cols + context_appraisals,
    "Gemma + Ranked appraisals": GEMMA_binary_cols + Ranked_cols,
    "Gemma + LDA appraisals": GEMMA_binary_cols + LDA_cols,
}

# ---------------------------------------------
# PHASE 1 Hyperparameter tuning on the first feature set
# ---------------------------------------------
first_feat_name, first_cols = next(iter(model_versions.items()))
X_tune = prosocial_train[first_cols].values
y_tune = prosocial_train.safety_label.map(BINARY_SAFETY_ORDER).values

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
    X = prosocial_train[cols].values
    y = prosocial_train.safety_label.map(BINARY_SAFETY_ORDER).values

    # Cross-validation on train set
    cv_scores = evaluate_classifier(best_clf_tuned, X, y, cv)

    # Refit on full train set, then evaluate on hold-out sets
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(best_clf_tuned))])
    pipe.fit(X, y)

    for test_name, test_df in [("valid",      prosocial_valid),
                                ("test", prosocial_test),
                                ("filtered", prosocial_test_filtered)]:
        X_test = test_df[cols].values
        y_test = test_df.safety_label.map(BINARY_SAFETY_ORDER).values
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
        pd.DataFrame({"label": y_test, "preds": preds}).to_csv(f"prosocialdialog_{test_name}_{best_clf_name}_{feat_name.replace(' ','').replace('+','')}_binary.csv.gz", index=False)

# ---------------------------------------------
# Save outputs
# ---------------------------------------------
results_df = pd.DataFrame(results)
tuning_df  = pd.DataFrame(tuning_results)

results_df.to_csv("prosocialdialog_results_binary_tuned.csv.gz", index=False)
tuning_df.to_csv("prosocialdialog_binary_tuning_summary.csv",     index=False)

print("\nTuning summary:")
print(tuning_df.to_string(index=False))
print("\nHold-out results (macro F1 + CV macro F1 mean + std):")
display_cols = ["Classifier", "Features", "Test set", "macro F1",
                "cv_f1_macro_mean", "cv_f1_macro_std", "Accuracy"]
print(results_df[display_cols].to_string(index=False))
print("\nDone. Results saved to prosocialdialog_results_binary_tuned.csv.gz and prosocialdialog_binary_tuning_summary.csv")


############ reduce dataset to only "__casual__" and "__needs_intervention__" labels
prosocial_train = prosocial_train[prosocial_train.safety_label.isin(["__casual__", "__needs_intervention__"])].reset_index()
prosocial_valid = prosocial_valid[prosocial_valid.safety_label.isin(["__casual__", "__needs_intervention__"])].reset_index()
prosocial_test = prosocial_test[prosocial_test.safety_label.isin(["__casual__", "__needs_intervention__"])].reset_index()
prosocial_test_filtered = prosocial_test[~prosocial_test['context'].isin(prosocial_train['context']) & ~prosocial_test['context'].isin(prosocial_valid['context'])].reset_index()

# ---------------------------------------------
# PHASE 1 Hyperparameter tuning on the first feature set
# ---------------------------------------------
first_feat_name, first_cols = next(iter(model_versions.items()))
X_tune = prosocial_train[first_cols].values
y_tune = prosocial_train.safety_label.map(BINARY_SAFETY_ORDER).values

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
    X = prosocial_train[cols].values
    y = prosocial_train.safety_label.map(BINARY_SAFETY_ORDER).values

    # Cross-validation on train set
    cv_scores = evaluate_classifier(best_clf_tuned, X, y, cv)

    # Refit on full train set, then evaluate on hold-out sets
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(best_clf_tuned))])
    pipe.fit(X, y)

    for test_name, test_df in [("valid",      prosocial_valid),
                                ("test", prosocial_test),
                                ("filtered", prosocial_test_filtered)]:
        X_test = test_df[cols].values
        y_test = test_df.safety_label.map(BINARY_SAFETY_ORDER).values
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
        pd.DataFrame({"label": y_test, "preds": preds}).to_csv(f"prosocialdialog_{test_name}_{best_clf_name}_{feat_name.replace(' ','').replace('+','')}_binary-reduced.csv.gz", index=False)

# ---------------------------------------------
# Save outputs
# ---------------------------------------------
results_df = pd.DataFrame(results)
tuning_df  = pd.DataFrame(tuning_results)

results_df.to_csv("prosocialdialog_results_binary-reduced_tuned.csv.gz", index=False)
tuning_df.to_csv("prosocialdialog_binary-reduced_tuning_summary.csv",     index=False)

print("\nTuning summary:")
print(tuning_df.to_string(index=False))
print("\nHold-out results (macro F1 + CV macro F1 mean + std):")
display_cols = ["Classifier", "Features", "Test set", "macro F1",
                "cv_f1_macro_mean", "cv_f1_macro_std", "Accuracy"]
print(results_df[display_cols].to_string(index=False))
print("\nDone. Results saved to prosocialdialog_results_binary-reduced_tuned.csv.gz and prosocialdialog_binary-reduced_tuning_summary.csv")
