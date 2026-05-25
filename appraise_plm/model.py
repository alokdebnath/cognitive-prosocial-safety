"""
Appraise-ATN: Attention-based Multitask Appraisal Regression Model

This module implements a neural architecture for predicting 21 appraisal dimensions from text.
The model uses a combination of cross-attention (for inter-dimension learning) and 
dimension-specific self-attention layers, followed by individual regression heads.

Architecture:
    Input Text → DeBERTa Embeddings → Cross-Attention → 21 Self-Attention Modules → 21 Regressors
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple

# Import loss functions from utils
try:
    from utils import (
        AdaptiveWeightedMSELoss,
        StandardSummationMSELoss,
        MahalanobisLoss,
        MaximumMeanDiscrepancyLoss,
        EarthMoversDistanceLoss,
        ConcordanceCorrelationCoefficientLoss,
        CosineSimilarityLoss
    )
except ImportError:
    from appraisals.utils import (
        AdaptiveWeightedMSELoss,
        StandardSummationMSELoss,
        MahalanobisLoss,
        MaximumMeanDiscrepancyLoss,
        EarthMoversDistanceLoss,
        ConcordanceCorrelationCoefficientLoss,
        CosineSimilarityLoss
    )


class DimensionSelfAttention(nn.Module):
    """
    Single self-attention module for one appraisal dimension.
    
    This module takes sequence embeddings and applies self-attention to extract
    dimension-specific features, then pools to a single vector.
    """
    
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Attention projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled output [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        Q = self.query(x)  # [batch, seq_len, hidden_dim]
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [batch, seq_len, seq_len]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len]
            # Expand to [batch, 1, seq_len] for broadcasting over query positions
            mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, seq_len, hidden_dim]
        
        # Residual connection and layer norm
        x = self.layer_norm(x + attn_output)
        
        # Mean pooling over sequence length (ignore padding)
        if attention_mask is not None:
            # attention_mask shape: [batch, seq_len]
            # x shape: [batch, seq_len, hidden_dim]
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()  # [batch, seq_len, hidden_dim]
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)  # [batch, hidden_dim]
            sum_mask = mask_expanded.sum(dim=1)  # [batch, hidden_dim]
            pooled = sum_embeddings / torch.clamp(sum_mask, min=1e-9)  # [batch, hidden_dim]
        else:
            pooled = x.mean(dim=1)  # [batch, hidden_dim]
            
        return pooled


class AppraiseATN(nn.Module):
    """
    Main multitask appraisal regression model.
    
    This model predicts 21 appraisal dimensions from text using:
    1. DeBERTa embeddings
    2. Cross-attention layer for inter-dimension relationships
    3. 21 parallel dimension-specific self-attention modules
    4. 21 independent regression heads
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/deberta-v3-small',
        num_appraisals: int = 21,
        num_cross_attn_heads: int = 6,
        freeze_embeddings: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_appraisals = num_appraisals
        self.model_name = model_name
        
        # 1. DeBERTa embeddings
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.hidden_dim = self.encoder.config.hidden_size
        
        if freeze_embeddings:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 2. Cross-attention layer (learns inter-dimension relationships)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(self.hidden_dim)
        
        # 3. Dimension-specific self-attention modules (21 parallel)
        self.dimension_attentions = nn.ModuleList([
            DimensionSelfAttention(self.hidden_dim, dropout)
            for _ in range(num_appraisals)
        ])
        
        # 4. Regression heads (21 independent)
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            )
            for _ in range(num_appraisals)
        ])
        
        # Learnable uncertainty parameters for each dimension
        self.log_vars = nn.Parameter(torch.zeros(num_appraisals))
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            return_attentions: Whether to return attention weights
            
        Returns:
            Dictionary with:
                - predictions: Appraisal predictions [batch_size, 21]
                - log_vars: Learnable uncertainty parameters [21]
        """
        # 1. Get DeBERTa embeddings
        # Pass token_type_ids if provided (DeBERTa v3 doesn't typically use them, but older/other models might)
        encoder_args = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None:
            encoder_args['token_type_ids'] = token_type_ids
            
        encoder_output = self.encoder(**encoder_args)
        # Cast to float32: DeBERTa-v3 can emit FP16 hidden states, but the
        # downstream cross-attention and self-attention layers are FP32.
        hidden_states = encoder_output.last_hidden_state.float()  # [batch, seq_len, hidden_dim]
        
        # 2. Cross-attention for inter-dimension learning
        cross_attn_output, cross_attn_weights = self.cross_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=(attention_mask == 0)
        )
        # Residual connection + layer norm
        hidden_states = self.cross_attn_norm(hidden_states + cross_attn_output)
        
        # 3. Apply dimension-specific self-attention (parallel processing)
        dimension_features = []
        for dim_attention in self.dimension_attentions:
            dim_feature = dim_attention(hidden_states, attention_mask)
            dimension_features.append(dim_feature)
        
        # Stack dimension features: [batch, num_appraisals, hidden_dim]
        dimension_features = torch.stack(dimension_features, dim=1)
        
        # 4. Apply regression heads
        predictions = []
        for i, regression_head in enumerate(self.regression_heads):
            logit = regression_head(dimension_features[:, i, :])  # [batch, 1]
            # Scale to [1, 5] range using sigmoid
            pred = 1.0 + 4.0 * torch.sigmoid(logit)
            predictions.append(pred.squeeze(-1))  # Squeeze to [batch] instead of [batch, 1]
        
        predictions = torch.stack(predictions, dim=-1)  # [batch, 21]
        
        output = {
            'predictions': predictions,
            'log_vars': self.log_vars
        }
        
        if return_attentions:
            output['cross_attn_weights'] = cross_attn_weights
            
        return output
    
    def save_pretrained(self, save_directory: str):
        """
        Save model in HuggingFace-compatible format.
        
        Args:
            save_directory: Directory to save model and config
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save encoder (DeBERTa)
        encoder_dir = os.path.join(save_directory, 'encoder')
        self.encoder.save_pretrained(encoder_dir)
        
        # Save tokenizer
        tokenizer = self.get_tokenizer()
        tokenizer.save_pretrained(encoder_dir)
        
        # Save full model state
        model_state = {
            'model_state_dict': self.state_dict(),
            'config': {
                'model_name': self.model_name,
                'num_appraisals': self.num_appraisals,
                'hidden_dim': self.hidden_dim,
                'num_cross_attn_heads': self.cross_attention.num_heads,
                'appraisal_dimensions': APPRAISAL_DIMENSIONS
            }
        }
        torch.save(model_state, os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save config as JSON for easy inspection
        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(model_state['config'], f, indent=2)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str, device: str = 'cpu'):
        """
        Load model from HuggingFace-compatible format.
        
        Args:
            model_directory: Directory containing saved model
            device: Device to load model to
            
        Returns:
            Loaded AppraiseATN model
        """
        import json
        
        # Load config
        config_path = os.path.join(model_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(
            model_name=config['model_name'],
            num_appraisals=config['num_appraisals'],
            num_cross_attn_heads=config.get('num_cross_attn_heads', 6)
        )
        
        # Load state dict
        state_dict_path = os.path.join(model_directory, 'pytorch_model.bin')
        checkpoint = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {model_directory}")
        return model
    
    def get_tokenizer(self):
        """Get the tokenizer for this model."""
        # Load tokenizer with explicit settings to avoid conversion issues
        try:
            # First try with use_fast=False
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=False,
                trust_remote_code=True
            )
        except Exception as e:
            # Fallback: try loading directly from DebertaV2Tokenizer
            print(f"Warning: AutoTokenizer failed ({e}), trying DebertaV2Tokenizer directly")
            from transformers import DebertaV2Tokenizer
            tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        
        return tokenizer





# Appraisal dimension names (for reference)
APPRAISAL_DIMENSIONS = [
    'suddenness', 'familiarity', 'predict_event', 
    'pleasantness', 'unpleasantness', 'goal_relevance',
    'chance_responsblt', 'self_responsblt', 'other_responsblt',
    'predict_conseq', 'goal_support', 'urgency',
    'self_control', 'other_control', 'chance_control',
    'accept_conseq', 'standards', 'social_norms',
    'attention', 'not_consider', 'effort'
]


if __name__ == '__main__':
    # Test model instantiation
    print("Testing Appraise-ATN model...")
    
    model = AppraiseATN(
        model_name='microsoft/deberta-v3-small',
        num_appraisals=21,
        num_cross_attn_heads=6
    )
    
    print(f"\nModel architecture:")
    print(f"  Encoder: {model.model_name}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num appraisals: {model.num_appraisals}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    tokenizer = model.get_tokenizer()
    test_text = ["This is a test sentence.", "Another example for testing."]
    
    inputs = tokenizer(
        test_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(**inputs)
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Predictions shape: {output['predictions'].shape}")
    print(f"  Predictions range: [{output['predictions'].min():.2f}, {output['predictions'].max():.2f}]")
    print(f"  Log vars shape: {output['log_vars'].shape}")
    
    # Test loss function
    print("\nTesting loss function...")
    dimension_vars = torch.rand(21) + 0.5  # Dummy variances
    loss_fn = AdaptiveWeightedMSELoss(dimension_vars)
    
    dummy_targets = torch.rand(2, 21) * 4 + 1  # Random targets in [1, 5]
    total_loss, loss_dict = loss_fn(output['predictions'], dummy_targets, output['log_vars'])
    
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  MSE loss: {loss_dict['mse_loss']:.4f}")
    
    print("\n✓ Model test completed successfully!")
