"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Forecasting

Research Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon 
Time Series Forecasting" (Lim et al., Google, 2021)
https://arxiv.org/abs/1912.09363

Why 99% of AI/ML Engineers Don't Know This:
1. Published in 2021 (relatively recent)
2. Complex architecture with many specialized components
3. Not in standard ML libraries (only PyTorch Forecasting)
4. Requires deep understanding of attention mechanisms
5. Combines multiple novel techniques (variable selection, quantile regression)

Key Features:
- Variable selection networks (identify important features)
- Temporal processing with LSTM + multi-head attention
- Quantile regression for uncertainty estimation
- Interpretable attention weights
- Handles static, known, and unknown future inputs
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class TFTConfig:
    """Configuration for Temporal Fusion Transformer."""
    
    # Data dimensions
    static_input_size: int = 10
    known_input_size: int = 5
    unknown_input_size: int = 5
    output_size: int = 1
    
    # Sequence lengths
    encoder_length: int = 24
    decoder_length: int = 6
    
    # Model architecture
    hidden_size: int = 128
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    
    # Quantile regression
    quantiles: List[float] = None
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    
    Combines:
    - Skip connection (residual)
    - Gating mechanism (GLU)
    - Layer normalization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Primary layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        
        # Context integration (if provided)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        
        # Output layers
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.output_fc = nn.Linear(hidden_size, output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip_fc = nn.Linear(input_size, output_size)
        else:
            self.skip_fc = None
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GRN.
        
        Args:
            x: Input tensor [batch, ..., input_size]
            context: Optional context tensor [batch, ..., context_size]
        
        Returns:
            Output tensor [batch, ..., output_size]
        """
        # Primary processing
        hidden = self.elu(self.fc1(x))
        
        # Add context if provided
        if context is not None and self.context_size is not None:
            hidden = hidden + self.context_fc(context)
        
        # Second layer
        hidden = self.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        
        # Gating mechanism (GLU)
        gate = torch.sigmoid(self.gate(hidden))
        output = self.output_fc(hidden)
        gated_output = gate * output
        
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
        
        # Residual + normalization
        output = self.layer_norm(skip + gated_output)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network - Learns which features are important.
    
    Uses GRNs to compute feature importance weights and select
    relevant features for prediction.
    """
    
    def __init__(
        self,
        input_sizes: List[int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)
        self.hidden_size = hidden_size
        
        # Individual GRNs for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout
            )
            for size in input_sizes
        ])
        
        # Flattened input size
        flattened_size = sum(input_sizes)
        
        # Variable selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size=flattened_size,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout=dropout,
            context_size=context_size
        )
    
    def forward(
        self,
        variables: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select important variables.
        
        Args:
            variables: List of variable tensors
            context: Optional context for selection
        
        Returns:
            Tuple of (selected_features, selection_weights)
        """
        # Process each variable through its GRN
        processed = [
            grn(var) for grn, var in zip(self.variable_grns, variables)
        ]
        
        # Flatten for selection
        flattened = torch.cat(variables, dim=-1)
        
        # Compute selection weights
        weights = self.selection_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)
        
        # Weighted combination
        stacked = torch.stack(processed, dim=-1)  # [..., hidden, num_vars]
        weights_expanded = weights.unsqueeze(-2)  # [..., 1, num_vars]
        selected = torch.sum(stacked * weights_expanded, dim=-1)
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretability.
    
    Standard multi-head attention but returns attention weights
    for interpretability.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Optional attention mask
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, q_len, self.embed_dim)
        output = self.out_proj(output)
        
        # Average attention weights across heads for interpretability
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return output, attn_weights_avg


class TemporalFusionTransformer(nn.Module):
    """
    Complete Temporal Fusion Transformer for multi-horizon forecasting.
    
    Architecture:
    1. Variable selection for static, known, unknown inputs
    2. LSTM encoder for temporal processing
    3. Multi-head attention for long-range dependencies
    4. Quantile regression for uncertainty estimation
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Variable selection networks
        self.static_vsn = VariableSelectionNetwork(
            input_sizes=[1] * config.static_input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        
        self.encoder_vsn = VariableSelectionNetwork(
            input_sizes=[1] * (config.known_input_size + config.unknown_input_size),
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size
        )
        
        self.decoder_vsn = VariableSelectionNetwork(
            input_sizes=[1] * config.known_input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size
        )
        
        # LSTM for temporal processing
        self.encoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.decoder_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Gated residual connection
        self.post_lstm_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )
        
        # Post-attention processing
        self.post_attention_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
            dropout=config.dropout
        )
        
        # Output layers (one per quantile)
        self.output_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.output_size)
            for _ in config.quantiles
        ])
    
    def forward(
        self,
        static_inputs: torch.Tensor,
        encoder_known: torch.Tensor,
        encoder_unknown: torch.Tensor,
        decoder_known: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.
        
        Args:
            static_inputs: Static features [batch, static_size]
            encoder_known: Known past features [batch, enc_len, known_size]
            encoder_unknown: Unknown past features [batch, enc_len, unknown_size]
            decoder_known: Known future features [batch, dec_len, known_size]
        
        Returns:
            Dictionary with predictions and attention weights
        """
        batch_size = static_inputs.shape[0]
        
        # 1. Static variable selection
        static_vars = [static_inputs[:, i:i+1] for i in range(self.config.static_input_size)]
        static_context, static_weights = self.static_vsn(static_vars)
        
        # 2. Encoder variable selection
        encoder_inputs = torch.cat([encoder_known, encoder_unknown], dim=-1)
        encoder_vars = [encoder_inputs[:, :, i:i+1] for i in range(encoder_inputs.shape[-1])]
        encoder_features, encoder_weights = self.encoder_vsn(encoder_vars, static_context.unsqueeze(1))
        
        # 3. Decoder variable selection
        decoder_vars = [decoder_known[:, :, i:i+1] for i in range(self.config.known_input_size)]
        decoder_features, decoder_weights = self.decoder_vsn(decoder_vars, static_context.unsqueeze(1))
        
        # 4. LSTM encoding
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_features)
        
        # 5. LSTM decoding
        decoder_output, _ = self.decoder_lstm(decoder_features, (hidden, cell))
        
        # 6. Gated residual connection
        lstm_output = self.post_lstm_grn(decoder_output)
        
        # 7. Multi-head attention
        # Use encoder output as keys/values, decoder as queries
        attn_output, attn_weights = self.attention(
            query=lstm_output,
            key=encoder_output,
            value=encoder_output
        )
        
        # 8. Post-attention processing
        output = self.post_attention_grn(attn_output)
        
        # 9. Quantile predictions
        predictions = {}
        for i, quantile in enumerate(self.config.quantiles):
            pred = self.output_layers[i](output)
            predictions[f'q{int(quantile*100)}'] = pred
        
        return {
            'predictions': predictions,
            'attention_weights': attn_weights,
            'static_weights': static_weights,
            'encoder_weights': encoder_weights,
            'decoder_weights': decoder_weights
        }


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic forecasting.
    
    Loss = max(q * (y - y_pred), (q - 1) * (y - y_pred))
    """
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: Dict of quantile predictions
            targets: Ground truth [batch, seq_len, output_size]
        
        Returns:
            Total quantile loss
        """
        losses = []
        
        for quantile in self.quantiles:
            key = f'q{int(quantile*100)}'
            pred = predictions[key]
            
            # Quantile loss
            error = targets - pred
            loss = torch.max(
                quantile * error,
                (quantile - 1) * error
            )
            losses.append(loss.mean())
        
        return torch.stack(losses).mean()


class TFTTrainer:
    """Trainer for Temporal Fusion Transformer."""
    
    def __init__(
        self,
        model: TemporalFusionTransformer,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = QuantileLoss(model.config.quantiles)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
    
    def train_step(
        self,
        static: torch.Tensor,
        encoder_known: torch.Tensor,
        encoder_unknown: torch.Tensor,
        decoder_known: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Single training step."""
        self.model.train()
        
        # Move to device
        static = static.to(self.device)
        encoder_known = encoder_known.to(self.device)
        encoder_unknown = encoder_unknown.to(self.device)
        decoder_known = decoder_known.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(static, encoder_known, encoder_unknown, decoder_known)
        
        # Compute loss
        loss = self.criterion(outputs['predictions'], targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        static: torch.Tensor,
        encoder_known: torch.Tensor,
        encoder_unknown: torch.Tensor,
        decoder_known: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Evaluate model."""
        self.model.eval()
        
        static = static.to(self.device)
        encoder_known = encoder_known.to(self.device)
        encoder_unknown = encoder_unknown.to(self.device)
        decoder_known = decoder_known.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(static, encoder_known, encoder_unknown, decoder_known)
            loss = self.criterion(outputs['predictions'], targets)
        
        return loss.item(), outputs


# Example usage
if __name__ == "__main__":
    # Configuration
    config = TFTConfig(
        static_input_size=5,
        known_input_size=3,
        unknown_input_size=4,
        output_size=1,
        encoder_length=24,
        decoder_length=6,
        hidden_size=128,
        lstm_layers=2,
        attention_heads=4,
        quantiles=[0.1, 0.5, 0.9]
    )
    
    # Create model
    model = TemporalFusionTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TFTTrainer(model, learning_rate=1e-3)
    
    # Generate dummy data
    batch_size = 32
    static = torch.randn(batch_size, config.static_input_size)
    encoder_known = torch.randn(batch_size, config.encoder_length, config.known_input_size)
    encoder_unknown = torch.randn(batch_size, config.encoder_length, config.unknown_input_size)
    decoder_known = torch.randn(batch_size, config.decoder_length, config.known_input_size)
    targets = torch.randn(batch_size, config.decoder_length, config.output_size)
    
    # Training step
    loss = trainer.train_step(static, encoder_known, encoder_unknown, decoder_known, targets)
    print(f"Training loss: {loss:.4f}")
    
    # Evaluation
    val_loss, outputs = trainer.evaluate(static, encoder_known, encoder_unknown, decoder_known, targets)
    print(f"Validation loss: {val_loss:.4f}")
    
    # Check predictions
    for quantile in config.quantiles:
        key = f'q{int(quantile*100)}'
        pred = outputs['predictions'][key]
        print(f"Prediction {key} shape: {pred.shape}")
    
    # Check attention weights
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"Static variable importance: {outputs['static_weights'][0]}")
