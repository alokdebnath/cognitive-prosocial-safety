"""
Evaluation script for Appraise-ATN model.

This script evaluates the trained model on the test set and generates
detailed per-dimension performance metrics.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from model import AppraiseATN, APPRAISAL_DIMENSIONS
from train import CrowdEnVENTDataset, load_crowd_envent_data


def compute_metrics(predictions, targets, appraisal_dims=APPRAISAL_DIMENSIONS):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted values [N, 21]
        targets: Ground truth values [N, 21]
        
    Returns:
        DataFrame with per-dimension metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics = []
    for i, dim in enumerate(appraisal_dims):
        pred = predictions[:, i]
        target = targets[:, i]
        
        # Pearson correlation
        corr, p_value = pearsonr(pred, target)
        
        # MSE and MAE
        mse = ((pred - target) ** 2).mean()
        mae = np.abs(pred - target).mean()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        metrics.append({
            'dimension': dim,
            'correlation': corr,
            'p_value': p_value,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'target_mean': target.mean(),
            'target_std': target.std(),
            'pred_mean': pred.mean(),
            'pred_std': pred.std()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add average row
    avg_metrics = {
        'dimension': 'AVERAGE',
        'correlation': metrics_df['correlation'].mean(),
        'p_value': np.nan,
        'mse': metrics_df['mse'].mean(),
        'mae': metrics_df['mae'].mean(),
        'rmse': metrics_df['rmse'].mean(),
        'target_mean': metrics_df['target_mean'].mean(),
        'target_std': metrics_df['target_std'].mean(),
        'pred_mean': metrics_df['pred_mean'].mean(),
        'pred_std': metrics_df['pred_std'].mean()
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    return metrics_df


def plot_correlation_heatmap(predictions, targets, output_path, appraisal_dims=APPRAISAL_DIMENSIONS):
    """Plot correlation heatmap between predicted and actual values."""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Compute correlation matrix
    corr_matrix = np.zeros((len(appraisal_dims), len(appraisal_dims)))
    for i in range(len(appraisal_dims)):
        for j in range(len(appraisal_dims)):
            corr_matrix[i, j], _ = pearsonr(predictions[:, i], targets[:, j])
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=appraisal_dims,
        yticklabels=appraisal_dims,
        square=True
    )
    plt.title('Correlation: Predictions vs Ground Truth', fontsize=14, pad=20)
    plt.xlabel('Ground Truth Dimensions', fontsize=12)
    plt.ylabel('Predicted Dimensions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix


def plot_per_dimension_scatter(predictions, targets, output_dir, appraisal_dims=APPRAISAL_DIMENSIONS):
    """Plot scatter plots for each dimension."""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Create grid plot
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, dim in enumerate(appraisal_dims):
        ax = axes[i]
        pred = predictions[:, i]
        target = targets[:, i]
        
        # Scatter plot
        ax.scatter(target, pred, alpha=0.3, s=10)
        
        # Perfect prediction line
        ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect')
        
        # Compute correlation
        corr, _ = pearsonr(pred, target)
        
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Prediction', fontsize=10)
        ax.set_title(f'{dim}\n(ρ={corr:.3f})', fontsize=11)
        ax.set_xlim([0.5, 5.5])
        ax.set_ylim([0.5, 5.5])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Remove extra subplots
    for i in range(len(appraisal_dims), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        all_predictions.append(outputs['predictions'])
        all_targets.append(labels)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_targets


def main(args):
    """Main evaluation loop."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_crowd_envent_data(args.data_dir)
    
    # Prepare test data
    test_texts = test_df['generated_text'].tolist()
    test_labels = test_df[APPRAISAL_DIMENSIONS].values
    
    # Load model
    print("\nLoading model...")
    model = AppraiseATN.from_pretrained(args.model_dir, device=str(device))
    
    # Load training state for info
    training_state_path = os.path.join(args.model_dir, 'training_state.pt')
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device, weights_only=False)
        print(f"Loaded model from epoch {training_state['epoch']}")
        print(f"Best validation correlation: {training_state['best_val_corr']:.4f}")
        model_args = training_state['args']
    else:
        # Fallback for older checkpoints
        print("Training state not found, using default parameters")
        model_args = {'max_length': 512}
    
    # Create test dataset and dataloader
    tokenizer = model.get_tokenizer()
    test_dataset = CrowdEnVENTDataset(
        test_texts, test_labels, tokenizer, 
        model_args.get('max_length', 512)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    predictions, targets = evaluate(model, test_loader, device)
    
    # Compute metrics
    metrics_df = compute_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80)
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Saved metrics to {metrics_path}")
    
    # Generate visualizations
    if args.generate_plots:
        print("\nGenerating visualizations...")
        
        # Correlation heatmap
        heatmap_path = os.path.join(args.output_dir, 'correlation_heatmap.png')
        plot_correlation_heatmap(predictions, targets, heatmap_path)
        print(f"  ✓ Saved correlation heatmap to {heatmap_path}")
        
        # Scatter plots
        plot_per_dimension_scatter(predictions, targets, args.output_dir)
        print(f"  ✓ Saved scatter plots to {os.path.join(args.output_dir, 'scatter_plots.png')}")
    
    # Log to W&B
    if args.use_wandb:
        # Log metrics table
        wandb.log({'test_metrics': wandb.Table(dataframe=metrics_df)})
        
        # Log summary metrics
        avg_metrics = metrics_df[metrics_df['dimension'] == 'AVERAGE'].iloc[0]
        wandb.log({
            'test/avg_correlation': avg_metrics['correlation'],
            'test/avg_mse': avg_metrics['mse'],
            'test/avg_mae': avg_metrics['mae']
        })
        
        # Log per-dimension correlations
        for _, row in metrics_df.iterrows():
            if row['dimension'] != 'AVERAGE':
                wandb.log({f"test/corr_{row['dimension']}": row['correlation']})
        
        # Log visualizations
        if args.generate_plots:
            wandb.log({
                'correlation_heatmap': wandb.Image(heatmap_path),
                'scatter_plots': wandb.Image(os.path.join(args.output_dir, 'scatter_plots.png'))
            })
        
        wandb.finish()
    
    print("\n✓ Evaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Appraise-ATN model')
    
    # Data and model
    parser.add_argument('--data_dir', type=str, default='../data/crowd-envent',
                        help='Path to crowd-EnVENT data directory')
    parser.add_argument('--model_dir', type=str, default='appraisals/appraisal-analysis/model',
                        help='Directory containing trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='appraisals/appraisal-analysis/results',
                        help='Output directory for results')
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--generate_plots', action='store_true', default=True,
                        help='Generate visualization plots')
    
    # Infrastructure
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # W&B
    parser.add_argument('--use_wandb', action='store_true',
                        help='Log results to W&B')
    parser.add_argument('--wandb_project', type=str, default='appraise-atn',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default='evaluation',
                        help='W&B run name')
    
    args = parser.parse_args()
    
    main(args)
