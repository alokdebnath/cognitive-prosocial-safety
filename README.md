# Prosociality Appraisal Paper Submission

This repository contains the consolidated source code required to reproduce the core results of our prosociality appraisal framework, stripped of model weights and heavy data files for a clean submission.

## Repository Structure

- **`appraise_plm/`**
  The core Appraise-ATN model implementation. Contains the architecture (`model.py`), loss functions and utilities (`utils.py`), and the core training loop (`train.py`).
  
- **`train_appraise_plm.py`**
  The primary entry point to train the model. This is a thin wrapper that routes execution directly to the core PLM trainer.
  *Usage*: `python train_appraise_plm.py --data_dir /path/to/data --output_dir ./models`

- **`annotate_datasets.py`**
  Unified script to use the trained model to annotate dialogue datasets (ProsocialDialog, DiaSafety, BeaverTails, CoSafe). Computes appraisal dimensions for prompts, responses, or full dialogues.
  *Usage*: `python annotate_datasets.py --corpus prosocial --mode dialogue --output_file annotated_prosocial.csv`

- **`run_analyses.py`**
  Consolidated pipeline for all statistical analyses described in the paper. It automatically:
  1. Computes MANOVA for joint discriminative capacity across all 21 dimensions.
  2. Computes univariate Mann-Whitney $U$ statistics and rank-biserial correlations ($r_{rb}$) with Benjamini-Hochberg FDR correction.
  3. Conducts Random-Effects Meta-Analysis estimating pooled effects and $I^2$ heterogeneity.
  4. Computes SHAP feature importance using Random Forests.
  5. Computes individual effect sizes across safety types and sources.
  *Usage*: `python run_analyses.py --data_dir /path/to/data --output_dir my_results`

## Reproducibility

Ensure you have the required dependencies (such as `torch`, `transformers`, `statsmodels`, `scipy`, `pandas`, `shap`, etc.) installed.
All scripts are designed to be run from this root directory.

