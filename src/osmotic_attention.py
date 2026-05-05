"""
Osmotic Attention — Core Implementation
Inspired by osmosis: information flows from low-density to high-density token regions
Paper: "Osmotic Attention: Information-Gradient Driven Context Flow in Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OsmoticAttention(nn.Module):
    """
    Osmotic Attention mechanism.
    
    Extends standard scaled dot-product attention with:
    1. Information density estimation per token (entropy-based)
    2. Osmotic gradient computation between token pairs  
    3. Learned membrane permeability gate
    4. Per-head osmotic coupling coefficient lambda
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.scale = math.sqrt(self.head_dim)
        
        # Standard QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Information density projection (rho)
        self.rho_proj = nn.Linear(self.head_dim, self.head_dim)
        
        # Membrane permeability gate
        self.membrane_proj = nn.Linear(2 * self.head_dim, 1)
        
        # Per-head osmotic coupling coefficient (lambda)
        # One lambda per head — learnable
        self.lambda_h = nn.Parameter(torch.zeros(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights — lambda starts at 0 so model begins like standard attention"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.rho_proj.weight)
        nn.init.zeros_(self.lambda_h)  # Start as standard attention!
    
    def compute_information_density(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token information density (entropy).
        
        Args:
            hidden: [batch, seq_len, head_dim]
        Returns:
            rho: [batch, seq_len] — information density per token
        """
        # Project to get distribution
        logits = self.rho_proj(hidden)  
        p = F.softmax(logits, dim=-1)
        
        # Shannon entropy: H = -sum(p * log(p))
        rho = -(p * torch.log(p + self.eps)).sum(dim=-1)
        return rho  # [batch, seq_len]
    
    def compute_osmotic_gradient(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise osmotic gradient matrix.
        delta_pi[i,j] = rho[j] - rho[i]
        Positive = token j is more information-rich = context should flow toward j
        
        Args:
            rho: [batch, seq_len]
        Returns:
            delta_pi: [batch, seq_len, seq_len]
        """
        # Efficient broadcasting: rho_j - rho_i for all pairs
        rho_j = rho.unsqueeze(1)  # [batch, 1, seq_len]
        rho_i = rho.unsqueeze(2)  # [batch, seq_len, 1]
        delta_pi = rho_j - rho_i  # [batch, seq_len, seq_len]
        return delta_pi
    
    def compute_membrane_permeability(
        self, 
        h_i: torch.Tensor, 
        h_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute learned membrane permeability gate between token pairs.
        M[i,j] = sigmoid(W_m * [h_i || h_j])
        
        Args:
            h_i: [batch, seq_len, head_dim]
            h_j: [batch, seq_len, head_dim]
        Returns:
            M: [batch, seq_len, seq_len]
        """
        seq_len = h_i.size(1)
        
        # Expand for pairwise computation
        h_i_exp = h_i.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, n, n, d]
        h_j_exp = h_j.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, n, n, d]
        
        # Concatenate and project
        h_cat = torch.cat([h_i_exp, h_j_exp], dim=-1)  # [B, n, n, 2d]
        M = torch.sigmoid(self.membrane_proj(h_cat).squeeze(-1))  # [B, n, n]
        return M
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ):
        """
        Forward pass of Osmotic Attention.
        
        Args:
            query: [batch, seq_len, embed_dim]
            key:   [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            attn_mask: optional mask [seq_len, seq_len]
            key_padding_mask: optional padding mask [batch, seq_len]
        
        Returns:
            output: [batch, seq_len, embed_dim]
            attn_weights: [batch, num_heads, seq_len, seq_len]
        """
        B, N, _ = query.shape
        
        # 1. Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 2. Reshape to multi-head
        # [B, N, embed] -> [B, heads, N, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Standard attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: [B, heads, N, N]
        
        # 4. Compute osmotic terms per head
        osmotic_term = torch.zeros_like(attn_scores)
        
        for h in range(self.num_heads):
            # Hidden states for this head
            q_h = Q[:, h, :, :]  # [B, N, head_dim]
            k_h = K[:, h, :, :]  # [B, N, head_dim]
            
            # Information density
            rho = self.compute_information_density(q_h)  # [B, N]
            
            # Osmotic gradient
            delta_pi = self.compute_osmotic_gradient(rho)  # [B, N, N]
            
            # Membrane permeability
            M = self.compute_membrane_permeability(q_h, k_h)  # [B, N, N]
            
            # Osmotic contribution: M * lambda_h * delta_pi
            osmotic_term[:, h, :, :] = M * self.lambda_h[h] * delta_pi
        
        # 5. Add osmotic term to attention scores
        attn_scores = attn_scores + osmotic_term
        
        # 6. Apply masks if provided
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # 7. Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 8. Weighted sum over values
        output = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]
        
        # 9. Reshape back
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights