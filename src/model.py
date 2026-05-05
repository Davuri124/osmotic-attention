"""
Osmotic Transformer — Full Model
Integrates OsmoticAttention into a BERT-style encoder architecture
"""

import torch
import torch.nn as nn
from src.osmotic_attention import OsmoticAttention


class OsmoticTransformerLayer(nn.Module):
    """
    Single transformer encoder layer with OsmoticAttention.
    Follows standard Pre-LN architecture for training stability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Osmotic Attention (our contribution)
        self.self_attn = OsmoticAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer Norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN + Self Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)

        # Pre-LN + FFN + Residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x, attn_weights


class OsmoticTransformer(nn.Module):
    """
    Full Osmotic Transformer Encoder.
    Drop-in replacement for BERT encoder.
    """

    def __init__(
        self,
        vocab_size: int = 30522,      # BERT vocab size
        embed_dim: int = 768,          # BERT-base hidden size
        num_heads: int = 12,           # BERT-base heads
        num_layers: int = 12,          # BERT-base layers
        ffn_dim: int = 3072,           # BERT-base FFN dim
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_labels: int = 2,           # For classification tasks
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Token + Position Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(embed_dim)

        # Stack of Osmotic Transformer Layers
        self.layers = nn.ModuleList([
            OsmoticTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def get_position_ids(self, input_ids):
        seq_len = input_ids.size(1)
        return torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
            labels: [batch] — for classification loss

        Returns:
            dict with logits, loss (if labels provided), all attention weights
        """
        B, N = input_ids.shape

        # Embeddings
        position_ids = self.get_position_ids(input_ids)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        # Convert attention mask to key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # True where padding

        # Pass through all layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask=key_padding_mask)
            all_attn_weights.append(attn_weights)

        # CLS token representation for classification
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': x,
            'attention_weights': all_attn_weights,
        }

    def get_lambda_values(self):
        """Returns all lambda values across all layers — useful for analysis"""
        lambdas = []
        for i, layer in enumerate(self.layers):
            lambdas.append({
                f'layer_{i}': layer.self_attn.lambda_h.detach().cpu()
            })
        return lambdas