#!/usr/bin/env python3
"""
Consolidated Prosociality Appraisal Analysis Script.

Computes:
  - MANOVA on 21 appraisal dimensions simultaneously.
  - Univariate effect sizes (Mann-Whitney U and rank-biserial correlation, with BH correction).
  - Cross-dataset consistency via random-effects meta-analysis (pooled effects, I^2).
  - SHAP feature importance analysis.
  - Subtype/Source correlations and effect sizes.
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from sklearn.ensemble import RandomForestClassifier
import shap

warnings.filterwarnings('ignore')

APPRAISAL_DIMENSIONS = [
    'suddenness', 'familiarity', 'predict_event',
    'pleasantness', 'unpleasantness', 'goal_relevance',
    'chance_responsblt', 'self_responsblt', 'other_responsblt',
    'predict_conseq', 'goal_support', 'urgency',
    'self_control', 'other_control', 'chance_control',
    'accept_conseq', 'standards', 'social_norms',
    'attention', 'not_consider', 'effort'
]

# =============================================================================
# HARMONIZATION
# =============================================================================

def load_and_harmonize_prosocial(filepath, dialogue_filepath=None, max_samples=None):
    df = pd.read_csv(filepath)
    high_safety = ['__casual__', '__possibly_needs_caution__', '__probably_needs_caution__']
    low_safety = ['__needs_caution__', '__needs_intervention__']
    df = df[df['safety_label'].isin(high_safety + low_safety)].copy()
    df['safe_numeric'] = df['safety_label'].apply(lambda x: 1 if x in high_safety else 0)
    rename_dict = {f'context_{dim}': f'prompt_{dim}' for dim in APPRAISAL_DIMENSIONS}
    df.rename(columns=rename_dict, inplace=True)
    for dim in APPRAISAL_DIMENSIONS:
        if f'response_{dim}' in df.columns:
            df[f'delta_{dim}'] = df[f'response_{dim}'] - df[f'prompt_{dim}']
    
    if dialogue_filepath and os.path.exists(dialogue_filepath):
        diag_df = pd.read_csv(dialogue_filepath)
        diag_cols = [c for c in diag_df.columns if c.startswith('dialogue_')]
        if diag_cols and len(diag_df) == len(df):
            df = pd.concat([df.reset_index(drop=True), diag_df[diag_cols].reset_index(drop=True)], axis=1)
            
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df

def load_and_harmonize_beavertails(filepath, dialogue_filepath=None, max_samples=None):
    df = pd.read_csv(filepath)
    df['safe_numeric'] = df['is_safe'].astype(int)
    for dim in APPRAISAL_DIMENSIONS:
        if f'response_{dim}' in df.columns and f'prompt_{dim}' in df.columns:
            df[f'delta_{dim}'] = df[f'response_{dim}'] - df[f'prompt_{dim}']
            
    if dialogue_filepath and os.path.exists(dialogue_filepath):
        diag_df = pd.read_csv(dialogue_filepath)
        diag_cols = [c for c in diag_df.columns if c.startswith('dialogue_')]
        if diag_cols and len(diag_df) == len(df):
            df = pd.concat([df.reset_index(drop=True), diag_df[diag_cols].reset_index(drop=True)], axis=1)
            
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df

def load_and_harmonize_diasafety(filepath, split='train', dialogue_filepath=None, max_samples=None):
    df = pd.read_csv(filepath)
    df = df[df['split'] == split].copy()
    df['safe_numeric'] = df['label'].map({'Safe': 1, 'Unsafe': 0})
    for dim in APPRAISAL_DIMENSIONS:
        if f'response_{dim}' in df.columns and f'prompt_{dim}' in df.columns:
            df[f'delta_{dim}'] = df[f'response_{dim}'] - df[f'prompt_{dim}']
            
    if dialogue_filepath and os.path.exists(dialogue_filepath):
        diag_df = pd.read_csv(dialogue_filepath)
        if 'split' in diag_df.columns:
            diag_df = diag_df[diag_df['split'] == split].reset_index(drop=True)
        diag_cols = [c for c in diag_df.columns if c.startswith('dialogue_')]
        if diag_cols and len(diag_df) == len(df):
            df = pd.concat([df.reset_index(drop=True), diag_df[diag_cols].reset_index(drop=True)], axis=1)
            
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df

# =============================================================================
# STATISTICAL UTILS
# =============================================================================

def compute_rank_biserial(safe_vals, unsafe_vals):
    safe_vals = np.asarray(safe_vals, dtype=float)
    unsafe_vals = np.asarray(unsafe_vals, dtype=float)
    safe_vals = safe_vals[~np.isnan(safe_vals)]
    unsafe_vals = unsafe_vals[~np.isnan(unsafe_vals)]
    if len(safe_vals) < 2 or len(unsafe_vals) < 2:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(safe_vals, unsafe_vals, alternative='two-sided')
    n1, n2 = len(safe_vals), len(unsafe_vals)
    r_rb = 1.0 - (2.0 * U) / (n1 * n2)
    return r_rb, U, p

def compute_dersimonian_laird(effects, variances):
    k = len(effects)
    if k < 2: return np.nan, np.nan, np.nan, np.nan
    w = 1.0 / variances
    sum_w = np.sum(w)
    sum_wx = np.sum(w * effects)
    q_stat = np.sum(w * (effects - sum_wx / sum_w)**2)
    df = k - 1
    c = sum_w - np.sum(w**2) / sum_w
    tau2 = (q_stat - df) / c if c > 0 and q_stat > df else 0.0
    i_squared = max(0.0, 100 * (q_stat - df) / q_stat) if q_stat > 0 else 0.0
    w_star = 1.0 / (variances + tau2)
    pooled_effect = np.sum(w_star * effects) / np.sum(w_star)
    return pooled_effect, i_squared, q_stat, tau2

# =============================================================================
# ANALYSES
# =============================================================================

def run_manova(df, role, output_dir, dataset_name):
    print(f"[{dataset_name}] Running MANOVA on {role}...")
    cols = [f'{role}_{d}' for d in APPRAISAL_DIMENSIONS]
    sub = df[cols + ['safe_numeric']].dropna()
    rename = {f'{role}_{d}': f'c_{d}' for d in APPRAISAL_DIMENSIONS}
    sub = sub.rename(columns=rename)
    
    lhs = ' + '.join(f'c_{d}' for d in APPRAISAL_DIMENSIONS)
    formula = f'{lhs} ~ safe_numeric'
    
    mv = MANOVA.from_formula(formula, data=sub)
    result = mv.mv_test()
    
    records = []
    for effect_name, block in result.results.items():
        stat_df = block['stat']
        for stat_name in stat_df.index:
            row = stat_df.loc[stat_name]
            records.append({
                'dataset': dataset_name,
                'role': role,
                'effect': effect_name,
                'statistic': stat_name,
                'value': row.get('Value', np.nan),
                'F': row.get('F Value', np.nan),
                'p_value': row.get('Pr > F', np.nan)
            })
    manova_df = pd.DataFrame(records)
    manova_df.to_csv(os.path.join(output_dir, f'{dataset_name}_{role}_manova.csv'), index=False)

def run_univariate(df, role, output_dir, dataset_name):
    print(f"[{dataset_name}] Computing MW-U and Rank-Biserial for {role}...")
    safe_mask = df['safe_numeric'] == 1
    unsafe_mask = df['safe_numeric'] == 0
    
    records = []
    for dim in APPRAISAL_DIMENSIONS:
        col = f'{role}_{dim}'
        if col not in df.columns: continue
        
        safe_vals = df.loc[safe_mask, col].values
        unsafe_vals = df.loc[unsafe_mask, col].values
        r_rb, u_stat, p_raw = compute_rank_biserial(safe_vals, unsafe_vals)
        records.append({
            'dataset': dataset_name,
            'role': role,
            'dimension': dim,
            'r_rb': r_rb,
            'U': u_stat,
            'p_raw': p_raw
        })
    
    if not records: return pd.DataFrame()
    res_df = pd.DataFrame(records)
    
    # Benjamini-Hochberg correction
    valid = ~res_df['p_raw'].isna()
    if valid.any():
        _, p_adj, _, _ = multipletests(res_df.loc[valid, 'p_raw'], method='fdr_bh')
        res_df.loc[valid, 'p_bh'] = p_adj
    else:
        res_df['p_bh'] = np.nan
        
    res_df.to_csv(os.path.join(output_dir, f'{dataset_name}_{role}_univariate.csv'), index=False)
    return res_df

def run_shap_analysis(train_df, dataset_name, role, output_dir):
    print(f"[{dataset_name}] Running SHAP Analysis for {role}...")
    feature_cols = [f'{role}_{dim}' for dim in APPRAISAL_DIMENSIONS]
    if not all(c in train_df.columns for c in feature_cols):
        print(f"Skipping SHAP for {role} (columns missing)")
        return
        
    # Drop rows with NaNs
    sub = train_df[feature_cols + ['safe_numeric']].dropna()
    if len(sub) < 100: return
    
    X = sub[feature_cols].values
    y = sub['safe_numeric'].values
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    df_shap = pd.DataFrame({'Feature': feature_cols, 'Mean_Abs_SHAP': mean_abs_shap})
    df_shap.sort_values('Mean_Abs_SHAP', ascending=False, inplace=True)
    df_shap.to_csv(os.path.join(output_dir, f'{dataset_name}_{role}_shap.csv'), index=False)

def run_meta_analysis(all_univariate_results, output_dir):
    print("Running Cross-Dataset Meta-Analysis...")
    meta_records = []
    
    df_all = pd.concat(all_univariate_results, ignore_index=True)
    # Using 'response' role as primary for cross dataset consistency, or iterate through roles
    for role in df_all['role'].unique():
        sub_role = df_all[df_all['role'] == role]
        for dim in APPRAISAL_DIMENSIONS:
            sub_dim = sub_role[sub_role['dimension'] == dim]
            if len(sub_dim) < 2: continue
            
            effects = sub_dim['r_rb'].values
            # Rough variance approximation for meta-analysis (se ~ 1/sqrt(N))
            # Assuming N=20000 limit was applied
            variances = np.full_like(effects, 1.0 / 20000) 
            
            pe, i2, q, tau2 = compute_dersimonian_laird(effects, variances)
            meta_records.append({
                'role': role,
                'dimension': dim,
                'pooled_r_rb': pe,
                'I_squared': i2,
                'Q_stat': q,
                'Tau_squared': tau2
            })
            
    if meta_records:
        meta_df = pd.DataFrame(meta_records)
        meta_df.to_csv(os.path.join(output_dir, 'meta_analysis_results.csv'), index=False)

def run_subtype_effect_sizes(df, dataset_name, subtype_col, role, output_dir):
    print(f"[{dataset_name}] Computing Subtype Effect Sizes by {subtype_col} for {role}...")
    if subtype_col not in df.columns: return
    
    records = []
    for st in df[subtype_col].dropna().unique():
        sub = df[df[subtype_col] == st]
        safe_mask = sub['safe_numeric'] == 1
        unsafe_mask = sub['safe_numeric'] == 0
        
        for dim in APPRAISAL_DIMENSIONS:
            col = f'{role}_{dim}'
            if col not in sub.columns: continue
            
            safe_vals = sub.loc[safe_mask, col].values
            unsafe_vals = sub.loc[unsafe_mask, col].values
            r_rb, u_stat, p_raw = compute_rank_biserial(safe_vals, unsafe_vals)
            records.append({
                'subtype': st,
                'dimension': dim,
                'r_rb': r_rb,
                'p_raw': p_raw
            })
            
    if records:
        pd.DataFrame(records).to_csv(os.path.join(output_dir, f'{dataset_name}_{subtype_col}_{role}_effects.csv'), index=False)

# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main(args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading data...")
    # Optional dialogue paths
    pd_diag = f"{args.data_dir}/prosocial-dialog-appraised/prosocial_dialog_train_dialogue_appraised.csv"
    bt_diag = f"{args.data_dir}/beavertails/beavertails_330k_train_dialogue_appraised.csv"
    ds_diag = f"{args.data_dir}/diasafety/diasafety_dialogue_appraised.csv"
    
    datasets = {}
    if os.path.exists(f"{args.data_dir}/prosocial-dialog-appraised"):
        datasets['ProsocialDialog'] = load_and_harmonize_prosocial(
            f"{args.data_dir}/prosocial-dialog-appraised/prosocial_dialog_train_appraised.csv",
            dialogue_filepath=pd_diag, max_samples=20000
        )
    if os.path.exists(f"{args.data_dir}/beavertails"):
        datasets['BeaverTails'] = load_and_harmonize_beavertails(
            f"{args.data_dir}/beavertails/beavertails_330k_train_appraised.csv",
            dialogue_filepath=bt_diag, max_samples=20000
        )
    if os.path.exists(f"{args.data_dir}/diasafety"):
        datasets['DiaSafety'] = load_and_harmonize_diasafety(
            f"{args.data_dir}/diasafety/diasafety_all_appraised.csv",
            dialogue_filepath=ds_diag, max_samples=20000
        )
        
    all_univariate_results = []
    roles = ['prompt', 'response', 'delta', 'dialogue']
    
    for name, df in datasets.items():
        print(f"\nProcessing {name} (N={len(df)})")
        
        for role in roles:
            # Only process if columns exist
            if not any(c.startswith(f'{role}_') for c in df.columns): continue
            
            run_manova(df, role, out_dir, name)
            univ_df = run_univariate(df, role, out_dir, name)
            if not univ_df.empty:
                all_univariate_results.append(univ_df)
            run_shap_analysis(df, name, role, out_dir)
            
            # Subtype correlations
            if name == 'ProsocialDialog':
                run_subtype_effect_sizes(df, name, 'source', role, out_dir)
            elif name == 'DiaSafety':
                run_subtype_effect_sizes(df, name, 'category', role, out_dir)

    if all_univariate_results:
        run_meta_analysis(all_univariate_results, out_dir)
        
    print("\nAll analyses complete. Results saved in:", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    main(parser.parse_args())
