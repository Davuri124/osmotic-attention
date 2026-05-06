"""
OsmoticBERT — Osmotic Attention plugged into pretrained BERT
Surgical replacement of BertSelfAttention with OsmoticSelfAttention
Preserves all 110M pretrained weights, adds ~50K osmotic parameters

For NeurIPS/ICML submission:
"Osmotic Attention: Information-Gradient Driven Context Flow in Transformers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import (
    BertModel,
    BertPreTrainedModel,
    BertConfig,
)
from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
    BertPooler,
)


# ─────────────────────────────────────────────────────────────
# Core: Osmotic Self Attention
# Replaces BertSelfAttention exactly
# ─────────────────────────────────────────────────────────────

class OsmoticSelfAttention(nn.Module):
    """
    Drop-in replacement for BertSelfAttention.
    Adds osmotic information gradient to standard BERT attention.
    
    New parameters added per layer:
    - rho_proj:       [head_dim, head_dim] × num_heads  
    - membrane_proj:  [2 × head_dim, 1]   × num_heads
    - lambda_h:       [num_heads] scalar per head
    Total new params: ~50K for BERT-base
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible "
                f"by num_attention_heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Standard BERT projections — will load pretrained weights
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type

        # ── Osmotic parameters (NEW) ──────────────────────────
        self.eps = 1e-8

        # Information density projection per head
        self.rho_proj = nn.Linear(
            self.attention_head_size,
            self.attention_head_size,
            bias=False
        )

        # Membrane permeability gate per head
        self.membrane_proj_i = nn.Linear(
            self.attention_head_size, 1, bias=True
        )
        self.membrane_proj_j = nn.Linear(
            self.attention_head_size, 1, bias=False
        )

        # Per-head osmotic coupling (starts at 0 = standard BERT)
        self.lambda_h = nn.Parameter(
            torch.zeros(self.num_attention_heads)
        )

        self._init_osmotic_weights()

    def _init_osmotic_weights(self):
        """
        Initialize osmotic parameters carefully.
        lambda=0 means model starts identical to pretrained BERT.
        """
        nn.init.xavier_uniform_(self.rho_proj.weight)
        nn.init.xavier_uniform_(self.membrane_proj_i.weight)
        nn.init.xavier_uniform_(self.membrane_proj_j.weight)
        nn.init.zeros_(self.membrane_proj_i.bias)
        nn.init.zeros_(self.lambda_h)  # Critical: start as vanilla BERT

    def transpose_for_scores(self, x):
        """Reshape to [batch, heads, seq_len, head_dim]"""
        B, N, _ = x.size()
        x = x.view(B, N, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def compute_information_density(self, hidden):
        """
        Compute Shannon entropy of projected hidden states.
        rho_i = -sum(p * log(p))  where p = softmax(W_rho * h_i)
        
        Args:
            hidden: [batch, seq_len, head_dim]
        Returns:
            rho: [batch, seq_len]
        """
        logits = self.rho_proj(hidden)
        p = F.softmax(logits, dim=-1)
        rho = -(p * torch.log(p + self.eps)).sum(dim=-1)
        return rho

    def compute_osmotic_gradient(self, rho):
        """
        Pairwise osmotic gradient: delta_pi[i,j] = rho[j] - rho[i]
        
        Args:
            rho: [batch, seq_len]
        Returns:
            delta_pi: [batch, seq_len, seq_len]
        """
        rho_j = rho.unsqueeze(1)   # [B, 1, N]
        rho_i = rho.unsqueeze(2)   # [B, N, 1]
        return rho_j - rho_i       # [B, N, N]

    def compute_membrane(self, h_i, h_j):
        # Memory efficient: factorized membrane
        # Instead of [B,N,N,2D] → compute per-token scores and outer product
        # Reduces memory from O(N²D) to O(ND)
        m_i = self.membrane_proj_i(h_i)  # [B, N, 1]
        m_j = self.membrane_proj_j(h_j)  # [B, N, 1]
        M = torch.sigmoid(m_i + m_j.transpose(1, 2))  # [B, N, N]
        return M

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # ── Standard BERT QKV projections ─────────────────────
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))
        # All: [B, heads, N, head_dim]

        B, H, N, D = Q.shape

        # ── Standard attention scores ──────────────────────────
        scale = math.sqrt(D)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / scale
        # [B, heads, N, N]

        # ── Osmotic term per head ──────────────────────────────
        osmotic_term = torch.zeros_like(attn_scores)

        for h in range(H):
            q_h = Q[:, h, :, :]   # [B, N, D]
            k_h = K[:, h, :, :]   # [B, N, D]

            # Information density
            rho = self.compute_information_density(q_h)   # [B, N]

            # Osmotic gradient
            delta_pi = self.compute_osmotic_gradient(rho) # [B, N, N]

            # Membrane permeability
            M = self.compute_membrane(q_h, k_h)           # [B, N, N]

            # Combine: M * lambda_h * delta_pi
            osmotic_term[:, h] = M * self.lambda_h[h] * delta_pi

        # ── Add osmotic term ───────────────────────────────────
        attn_scores = attn_scores + osmotic_term

        # ── Apply attention mask (BERT style) ─────────────────
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ── Softmax + dropout ──────────────────────────────────
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ── Head mask ─────────────────────────────────────────
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        # ── Context layer ─────────────────────────────────────
        context = torch.matmul(attn_probs, V)              # [B, H, N, D]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(B, N, self.all_head_size)   # [B, N, hidden]

        outputs = (context, attn_probs) if output_attentions else (context,)
        return outputs


# ─────────────────────────────────────────────────────────────
# OsmoticBertAttention — wraps OsmoticSelfAttention + BertSelfOutput
# ─────────────────────────────────────────────────────────────

class OsmoticBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = OsmoticSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# ─────────────────────────────────────────────────────────────
# OsmoticBertLayer — full transformer layer with osmotic attention
# ─────────────────────────────────────────────────────────────

class OsmoticBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = OsmoticBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


# ─────────────────────────────────────────────────────────────
# OsmoticBertEncoder — stack of OsmoticBertLayers
# ─────────────────────────────────────────────────────────────

class OsmoticBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            OsmoticBertLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


# ─────────────────────────────────────────────────────────────
# OsmoticBertForSequenceClassification — full model for GLUE
# ─────────────────────────────────────────────────────────────

class OsmoticBertForSequenceClassification(nn.Module):
    """
    OsmoticBERT for sequence classification.
    Loads pretrained BERT weights, replaces attention with osmotic attention.
    Fine-tune on GLUE tasks.
    """

    def __init__(self, config, num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Standard BERT embeddings
        bert = BertModel(config)
        self.embeddings = bert.embeddings
        self.encoder = OsmoticBertEncoder(config)
        self.pooler = BertPooler(config)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self._init_classifier()

    def _init_classifier(self):
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def get_extended_attention_mask(self, attention_mask):
        """Convert [B, N] mask to [B, 1, 1, N] for broadcasting"""
        extended = attention_mask[:, None, None, :]
        extended = (1.0 - extended.float()) * -10000.0
        return extended

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
    ):
        extended_mask = None
        if attention_mask is not None:
            extended_mask = self.get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_mask,
            output_attentions=output_attentions,
        )

        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": encoder_outputs.get("hidden_states"),
            "attentions": encoder_outputs.get("attentions"),
        }

    def get_lambda_values(self):
        """Get all lambda values — for analysis and paper figures"""
        lambdas = {}
        for i, layer in enumerate(self.encoder.layer):
            for h, lam in enumerate(layer.attention.self.lambda_h):
                lambdas[f"layer{i}_head{h}"] = lam.item()
        return lambdas


# ─────────────────────────────────────────────────────────────
# Weight Loading — The Critical Function
# Loads pretrained BERT weights into OsmoticBERT
# ─────────────────────────────────────────────────────────────

def load_pretrained_bert_weights(osmotic_model, model_name="bert-base-uncased"):
    """
    Load pretrained BERT weights into OsmoticBERT.
    Maps BERT weight names to OsmoticBERT weight names.
    Osmotic parameters (lambda, membrane, rho) stay randomly initialized.
    """
    from transformers import BertModel
    print(f"Loading pretrained weights from {model_name}...")

    pretrained = BertModel.from_pretrained(model_name)
    pretrained_state = pretrained.state_dict()

    osmotic_state = osmotic_model.state_dict()
    loaded = 0
    skipped = 0

    for name, param in pretrained_state.items():
        # Map BERT names to our model names
        # BERT: encoder.layer.0.attention.self.query.weight
        # Ours: encoder.layer.0.attention.self.query.weight (same!)

        if name in osmotic_state:
            if osmotic_state[name].shape == param.shape:
                osmotic_state[name].copy_(param)
                loaded += 1
            else:
                print(f"  Shape mismatch: {name}")
                skipped += 1
        elif name.startswith("pooler"):
            # Map pooler weights
            mapped = name
            if mapped in osmotic_state:
                osmotic_state[mapped].copy_(param)
                loaded += 1
        else:
            skipped += 1

    osmotic_model.load_state_dict(osmotic_state)

    total_params = sum(p.numel() for p in osmotic_model.parameters())
    osmotic_params = sum(
        p.numel() for n, p in osmotic_model.named_parameters()
        if any(x in n for x in ["lambda_h", "membrane_proj", "rho_proj"])
    )

    print(f"✅ Loaded {loaded} weight tensors from pretrained BERT")
    print(f"⏭️  Skipped {skipped} tensors (osmotic params stay initialized)")
    print(f"📊 Total params: {total_params:,}")
    print(f"🧪 Osmotic params: {osmotic_params:,} ({100*osmotic_params/total_params:.2f}%)")

    del pretrained
    return osmotic_model


# ─────────────────────────────────────────────────────────────
# Factory function — create OsmoticBERT ready to fine-tune
# ─────────────────────────────────────────────────────────────

def create_osmotic_bert(num_labels=2, model_name="bert-base-uncased"):
    """
    Create OsmoticBERT with pretrained weights loaded.
    Ready to fine-tune on any classification task.
    """
    config = BertConfig.from_pretrained(model_name)
    model = OsmoticBertForSequenceClassification(config, num_labels=num_labels)
    model = load_pretrained_bert_weights(model, model_name)
    return model, config