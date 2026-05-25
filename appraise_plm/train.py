"""
Training script for Appraise-ATN model.

This script trains the multitask appraisal regression model on the crowd-EnVENT dataset
with adaptive weighted MSE loss and uncertainty weighting.
"""

import os
import sys

# Ensure the appraisals/ directory is on sys.path so 'model' and 'utils' resolve
# correctly regardless of whether the script is run from the repo root or from
# within the appraisals/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from model import AppraiseATN, AdaptiveWeightedMSELoss, APPRAISAL_DIMENSIONS
from utils import (
    AdaptiveWeightedMSELoss,
    StandardSummationMSELoss,
    MahalanobisLoss,
    MaximumMeanDiscrepancyLoss,
    EarthMoversDistanceLoss,
    ConcordanceCorrelationCoefficientLoss,
    CosineSimilarityLoss
)

# Available loss types
LOSS_TYPES = [
    'adaptive_mse',      # AdaptiveWeightedMSELoss (default)
    'standard_mse',      # StandardSummationMSELoss
    'mahalanobis',       # MahalanobisLoss
    'mmd',               # MaximumMeanDiscrepancyLoss
    'emd',               # EarthMoversDistanceLoss (Energy Distance)
    'ccc',               # ConcordanceCorrelationCoefficientLoss
    'cosine',            # CosineSimilarityLoss
]


class CrowdEnVENTDataset(Dataset):
    """Dataset class for crowd-EnVENT appraisal data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def load_crowd_envent_data(data_dir='data/crowd-envent'):
    """
    Load and preprocess crowd-EnVENT dataset.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_csv(os.path.join(data_dir, 'crowd-enVent-train.tsv'), sep='\t')
    val_df = pd.read_csv(os.path.join(data_dir, 'crowd-enVent-val.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(data_dir, 'crowd-enVent-test.tsv'), sep='\t')
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def compute_dimension_variances(train_df, appraisal_dims=APPRAISAL_DIMENSIONS):
    """
    Compute variance for each appraisal dimension.
    
    Args:
        train_df: Training dataframe
        appraisal_dims: List of appraisal dimension names
        
    Returns:
        Tensor of variances [num_appraisals]
    """
    variances = []
    for dim in appraisal_dims:
        var = train_df[dim].var()
        variances.append(var)
    
    variances = torch.tensor(variances, dtype=torch.float)
    
    print(f"\nDimension variances:")
    for dim, var in zip(appraisal_dims, variances):
        print(f"  {dim:20s}: {var:.4f}")
    
    return variances


def compute_inverse_covariance(train_df, appraisal_dims=APPRAISAL_DIMENSIONS):
    """
    Compute the inverse covariance matrix (precision matrix) for Mahalanobis loss.
    
    Args:
        train_df: Training dataframe
        appraisal_dims: List of appraisal dimension names
        
    Returns:
        Tensor of inverse covariance matrix [num_appraisals, num_appraisals]
    """
    # Extract appraisal values as numpy array
    data = train_df[appraisal_dims].values  # [N, 21]
    
    # Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)  # [21, 21]
    
    # Add small regularization for numerical stability
    cov_matrix += np.eye(len(appraisal_dims)) * 1e-6
    
    # Compute inverse
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    print(f"\nComputed inverse covariance matrix for Mahalanobis loss.")
    print(f"  Shape: {inv_cov_matrix.shape}")
    print(f"  Condition number: {np.linalg.cond(cov_matrix):.2f}")
    
    return torch.tensor(inv_cov_matrix, dtype=torch.float)


def create_loss_function(loss_type, device, dimension_variances=None, inverse_covariance=None):
    """
    Factory function to create the specified loss function.
    
    Args:
        loss_type: String identifier for the loss type
        device: torch device
        dimension_variances: Tensor of variances (for adaptive_mse)
        inverse_covariance: Inverse covariance matrix (for mahalanobis)
        
    Returns:
        Tuple of (loss_fn, requires_log_vars)
    """
    requires_log_vars = False
    
    if loss_type == 'adaptive_mse':
        if dimension_variances is None:
            raise ValueError("adaptive_mse requires dimension_variances")
        loss_fn = AdaptiveWeightedMSELoss(dimension_variances.to(device))
        requires_log_vars = True
        
    elif loss_type == 'standard_mse':
        loss_fn = StandardSummationMSELoss()
        
    elif loss_type == 'mahalanobis':
        if inverse_covariance is None:
            raise ValueError("mahalanobis requires inverse_covariance")
        loss_fn = MahalanobisLoss(inverse_covariance.to(device))
        
    elif loss_type == 'mmd':
        loss_fn = MaximumMeanDiscrepancyLoss(kernel_alphas=[0.1, 1.0, 10.0])
        
    elif loss_type == 'emd':
        loss_fn = EarthMoversDistanceLoss()
        
    elif loss_type == 'ccc':
        loss_fn = ConcordanceCorrelationCoefficientLoss()
        
    elif loss_type == 'cosine':
        loss_fn = CosineSimilarityLoss()
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {LOSS_TYPES}")
    
    print(f"\nUsing loss function: {loss_fn.__class__.__name__}")
    
    return loss_fn, requires_log_vars


def compute_correlations(predictions, targets, appraisal_dims=APPRAISAL_DIMENSIONS):
    """
    Compute per-dimension Pearson correlations.
    
    Args:
        predictions: Predicted values [batch_size, 21]
        targets: Ground truth values [batch_size, 21]
        appraisal_dims: List of dimension names
        
    Returns:
        Dictionary of correlations and average
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    correlations = {}
    for i, dim in enumerate(appraisal_dims):
        corr, _ = pearsonr(predictions[:, i], targets[:, i])
        correlations[dim] = corr
    
    correlations['average'] = np.mean(list(correlations.values()))
    
    return correlations


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch, requires_log_vars=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss (pass log_vars only if required)
        if requires_log_vars:
            loss, loss_dict = loss_fn(outputs['predictions'], labels, outputs['log_vars'])
        else:
            loss, loss_dict = loss_fn(outputs['predictions'], labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        all_predictions.append(outputs['predictions'].detach())
        all_targets.append(labels.detach())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mse': f"{loss_dict.get('mse_loss', loss_dict.get('total_loss', torch.tensor(0.0))).item():.4f}"
        })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    correlations = compute_correlations(all_predictions, all_targets)
    
    return avg_loss, correlations


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, requires_log_vars=True):
    """Validate on validation set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Validating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss (pass log_vars only if required)
        if requires_log_vars:
            loss, _ = loss_fn(outputs['predictions'], labels, outputs['log_vars'])
        else:
            loss, _ = loss_fn(outputs['predictions'], labels)
        
        total_loss += loss.item()
        all_predictions.append(outputs['predictions'])
        all_targets.append(labels)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    correlations = compute_correlations(all_predictions, all_targets)
    
    return avg_loss, correlations


def main(args):
    """Main training loop."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize W&B (optional)
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            tags=[f"loss:{args.loss_type}"]
        )
    else:
        print("\nW&B logging disabled.")
    
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_crowd_envent_data(args.data_dir)
    
    # Compute dimension variances for weighted loss
    dimension_variances = compute_dimension_variances(train_df)
    
    # Extract features and labels
    train_texts = train_df['generated_text'].tolist()
    train_labels = train_df[APPRAISAL_DIMENSIONS].values
    
    val_texts = val_df['generated_text'].tolist()
    val_labels = val_df[APPRAISAL_DIMENSIONS].values
    
    # Initialize model
    print("\nInitializing model...")
    model = AppraiseATN(
        model_name=args.model_name,
        num_appraisals=len(APPRAISAL_DIMENSIONS),
        num_cross_attn_heads=args.num_cross_attn_heads,
        freeze_embeddings=args.freeze_embeddings,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create datasets and dataloaders
    tokenizer = model.get_tokenizer()
    
    train_dataset = CrowdEnVENTDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = CrowdEnVENTDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Precompute statistics needed for loss functions
    inverse_covariance = None
    if args.loss_type == 'mahalanobis':
        inverse_covariance = compute_inverse_covariance(train_df)
    
    # Loss function
    loss_fn, requires_log_vars = create_loss_function(
        args.loss_type,
        device,
        dimension_variances=dimension_variances,
        inverse_covariance=inverse_covariance
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_corr = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    print(f"Warmup steps: {warmup_steps} / {total_steps}")
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_corrs = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, epoch, requires_log_vars
        )
        
        # Log training metrics
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/avg_correlation': train_corrs['average'],
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            # Log per-dimension correlations
            for dim, corr in train_corrs.items():
                if dim != 'average':
                    wandb.log({f'train/corr_{dim}': corr})
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Avg Corr: {train_corrs['average']:.4f}")
        
        # Validate every N epochs
        if epoch % args.val_every == 0:
            val_loss, val_corrs = validate(model, val_loader, loss_fn, device, requires_log_vars)
            
            # Log validation metrics
            if use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/avg_correlation': val_corrs['average']
                })
                for dim, corr in val_corrs.items():
                    if dim != 'average':
                        wandb.log({f'val/corr_{dim}': corr})
            
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Avg Corr: {val_corrs['average']:.4f}")
            
            # Determine if we should save the model based on checkpoint metric
            save_model = False
            improved_metrics = []
            degraded_metrics = []
            
            # Check correlation improvement
            corr_improved = val_corrs['average'] > best_val_corr
            if corr_improved:
                improved_metrics.append(f"corr: {best_val_corr:.4f} → {val_corrs['average']:.4f}")
            elif best_val_corr > -1.0:  # Not first epoch
                degraded_metrics.append(f"corr: {best_val_corr:.4f} → {val_corrs['average']:.4f}")
            
            # Check loss improvement
            loss_improved = val_loss < best_val_loss
            if loss_improved:
                improved_metrics.append(f"loss: {best_val_loss:.4f} → {val_loss:.4f}")
            elif best_val_loss < float('inf'):  # Not first epoch
                degraded_metrics.append(f"loss: {best_val_loss:.4f} → {val_loss:.4f}")
            
            # Decide whether to save based on checkpoint_metric setting
            if args.checkpoint_metric == 'correlation':
                save_model = corr_improved
                metric_for_patience = corr_improved
            elif args.checkpoint_metric == 'loss':
                save_model = loss_improved
                metric_for_patience = loss_improved
            elif args.checkpoint_metric == 'both':
                save_model = corr_improved and loss_improved
                metric_for_patience = save_model
            
            # Print metric changes
            if improved_metrics:
                print(f"  Improved: {', '.join(improved_metrics)}")
            if degraded_metrics:
                print(f"  Degraded: {', '.join(degraded_metrics)}")
            
            # Save model if criteria met
            if save_model:
                best_val_corr = val_corrs['average']
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model in HuggingFace format
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                
                # Also save training checkpoint
                torch.save({
                    'epoch': epoch,
                    'best_val_corr': best_val_corr,
                    'best_val_loss': best_val_loss,
                    'checkpoint_metric': args.checkpoint_metric,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'args': vars(args)
                }, os.path.join(args.output_dir, 'training_state.pt'))
                
                print(f"  ✓ Saved best model (corr: {best_val_corr:.4f}, loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Model not saved (checkpoint_metric='{args.checkpoint_metric}' not satisfied)")
                print(f"  Patience: {patience_counter}/{args.patience}")
            
            # Log best metrics and checkpoint status to W&B
            if use_wandb:
                wandb.log({
                    'best/val_loss': best_val_loss,
                    'best/val_correlation': best_val_corr,
                    'checkpoint/model_saved': 1 if save_model else 0,
                    'checkpoint/patience_counter': patience_counter,
                    'checkpoint/corr_improved': 1 if corr_improved else 0,
                    'checkpoint/loss_improved': 1 if loss_improved else 0,
                })
            
            # Early stopping based on the chosen metric
            if not metric_for_patience:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation - corr: {best_val_corr:.4f}, loss: {best_val_loss:.4f}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation - corr: {best_val_corr:.4f}, loss: {best_val_loss:.4f}")
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Appraise-ATN model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/crowd-envent',
                        help='Path to crowd-EnVENT data directory')
    parser.add_argument('--output_dir', type=str, default='appraisals/appraisal-analysis/model',
                        help='Output directory for model checkpoints')
    
    # Model
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small',
                        help='Pretrained model name')
    parser.add_argument('--num_cross_attn_heads', type=int, default=6,
                        help='Number of cross-attention heads')
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Freeze DeBERTa embeddings')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (fraction of total steps)')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint_metric', type=str, default='loss',
                        choices=['correlation', 'loss', 'both'],
                        help='Metric to use for model checkpointing: '
                             'correlation (save when val corr improves), '
                             'loss (save when val loss decreases - prevents overfitting), '
                             'both (save only when both metrics improve)')
    parser.add_argument('--loss_type', type=str, default='adaptive_mse',
                        choices=LOSS_TYPES,
                        help=f'Loss function to use. Available: {LOSS_TYPES}')
    
    # Infrastructure
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() or 2),
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # W&B
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging (run offline / without W&B)')
    parser.add_argument('--wandb_project', type=str, default='appraise-atn',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='W&B run name')
    
    args = parser.parse_args()
    
    # Set default run name to include loss_type for easy comparison
    if args.run_name is None:
        args.run_name = f"appraise-atn-{args.loss_type}-{args.model_name.split('/')[-1]}-bs{args.batch_size}"
    
    main(args)
