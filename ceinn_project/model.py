import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AblationFlags:
    use_pt: bool = True
    use_hd: bool = True
    use_mtl: bool = True
    use_causal: bool = True


class EconomicSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma=None, use_hd=True):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if use_hd and gamma is not None:
            # gamma: [B, T, 1] -> each query position has its own bias intensity
            positions = torch.arange(seq_len, device=x.device)
            delta = (positions.view(1, seq_len, 1) - positions.view(1, 1, seq_len)).clamp(min=0).float()
            gamma_q = gamma.squeeze(-1).unsqueeze(-1)  # [B,T,1]
            hyper_bias = -torch.log1p(gamma_q * delta + 1e-8)  # [B,T,T]
            hyper_bias = hyper_bias.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            scores = scores + hyper_bias.unsqueeze(1)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class EconomicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = EconomicSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma=None, use_hd=True):
        x = self.norm1(x + self.dropout(self.attn(x, gamma=gamma, use_hd=use_hd)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class CEINN(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_side_features,
        item_propensity,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        semantic_dim: int = 64,
        economic_dim: int = 64,
        num_confounder_prototypes: int = 8,
        tau_propensity_clip: float = 10.0,
        lambda_loss_aversion: float = 2.0,
        alpha_gain: float = 0.7,
        beta_loss: float = 0.8,
        gate_temperature: float = 0.5,
        kappa_init: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.tau_propensity_clip = tau_propensity_clip
        self.lambda_loss_aversion = lambda_loss_aversion
        self.alpha_gain = alpha_gain
        self.beta_loss = beta_loss
        self.gate_temperature = gate_temperature

        side_features = torch.tensor(item_side_features, dtype=torch.float32)
        self.register_buffer('item_side_features', side_features)
        self.register_buffer('item_propensity', torch.tensor(item_propensity, dtype=torch.float32))

        side_dim = side_features.shape[1]
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(512, d_model)
        self.item_side_proj = nn.Linear(side_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EconomicTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.semantic_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, semantic_dim)
        )
        self.economic_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, economic_dim)
        )

        self.gamma_head = nn.Linear(economic_dim, 1)
        self.reference_head = nn.Linear(economic_dim, 1)
        self.eta_head = nn.Linear(economic_dim + 1, 1)

        self.short_head = nn.Sequential(
            nn.Linear(semantic_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.do_item_proj = nn.Linear(d_model + economic_dim + d_model, d_model)
        self.do_out = nn.Linear(d_model, 1)

        self.confounder_prototypes = nn.Parameter(torch.randn(num_confounder_prototypes, d_model) * 0.02)
        self.confounder_logits = nn.Parameter(torch.zeros(num_confounder_prototypes))
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))

    def encode_sequence(self, seq, use_hd=True):
        device = seq.device
        bsz, seq_len = seq.shape
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.item_emb(seq) + self.pos_emb(pos_ids) + self.item_side_proj(self.item_side_features[seq])
        x = self.dropout(x)

        # preliminary economic state for dynamic gamma
        prelim = self.economic_mlp(x)
        gamma = torch.sigmoid(self.gamma_head(prelim))
        for block in self.blocks:
            x = block(x, gamma=gamma, use_hd=use_hd)
        x = self.final_norm(x)
        return x

    def get_last_hidden(self, seq_hidden, seq):
        lengths = (seq != 0).sum(dim=1).clamp(min=1)
        idx = lengths - 1
        return seq_hidden[torch.arange(seq.size(0), device=seq.device), idx]

    def dynamic_reference(self, economic_state, candidate_abs_util):
        # economic_state: [B, de], candidate_abs_util: [B, C]
        base_ref = self.reference_head(economic_state).squeeze(-1)  # [B]
        mean_abs = candidate_abs_util.mean(dim=1, keepdim=True)  # [B,1]
        eta_in = torch.cat([economic_state, mean_abs], dim=-1)
        eta = torch.sigmoid(self.eta_head(eta_in))  # [B,1]
        ref = eta.squeeze(-1) * mean_abs.squeeze(-1) + (1.0 - eta.squeeze(-1)) * base_ref
        return ref

    def smooth_prospect_value(self, delta, use_pt=True):
        if not use_pt:
            return delta
        gate = torch.sigmoid(delta / max(self.gate_temperature, 1e-6))
        eps = 1e-6
        gain = torch.pow(delta.pow(2) + eps, self.alpha_gain / 2.0)
        loss = torch.pow(delta.pow(2) + eps, self.beta_loss / 2.0)
        return gate * gain - self.lambda_loss_aversion * (1.0 - gate) * loss

    def do_intervention_utility(self, economic_state, candidate_items, use_causal=True):
        # candidate_items: [B, C]
        item_emb = self.item_emb(candidate_items) + self.item_side_proj(self.item_side_features[candidate_items])
        econ = economic_state.unsqueeze(1).expand(-1, candidate_items.size(1), -1)

        if use_causal:
            prior = F.softmax(self.confounder_logits, dim=0)  # [K]
            proto = self.confounder_prototypes.unsqueeze(0).unsqueeze(0)  # [1,1,K,D]
            item_e = item_emb.unsqueeze(2).expand(-1, -1, proto.size(2), -1)
            econ_e = econ.unsqueeze(2).expand_as(item_e)
            fusion = torch.cat([item_e, econ_e, proto.expand(item_e.size(0), item_e.size(1), -1, -1)], dim=-1)
            hidden = torch.tanh(self.do_item_proj(fusion))
            logits = self.do_out(hidden).squeeze(-1)  # [B,C,K]
            do_util = (logits * prior.view(1, 1, -1)).sum(dim=-1)  # [B,C]
        else:
            fusion = torch.cat([item_emb, econ], dim=-1)
            hidden = torch.tanh(self.do_item_proj(torch.cat([item_emb, econ, item_emb], dim=-1)))
            do_util = self.do_out(hidden).squeeze(-1)
        return do_util, item_emb

    def short_term_score(self, semantic_state, candidate_item_emb):
        sem = semantic_state.unsqueeze(1).expand(-1, candidate_item_emb.size(1), -1)
        return self.short_head(torch.cat([sem, candidate_item_emb], dim=-1)).squeeze(-1)

    def forward(self, seq, candidate_items, flags: AblationFlags):
        seq_hidden = self.encode_sequence(seq, use_hd=flags.use_hd)
        last_hidden = self.get_last_hidden(seq_hidden, seq)
        semantic_state = self.semantic_mlp(last_hidden)
        economic_state = self.economic_mlp(last_hidden)

        do_abs_util, candidate_item_emb = self.do_intervention_utility(
            economic_state, candidate_items, use_causal=flags.use_causal
        )
        ref = self.dynamic_reference(economic_state, do_abs_util)
        delta = do_abs_util - ref.unsqueeze(-1)
        long_score = self.smooth_prospect_value(delta, use_pt=flags.use_pt)

        short_score = self.short_term_score(semantic_state, candidate_item_emb)
        if not flags.use_mtl:
            short_score = torch.zeros_like(short_score)

        total_score = short_score + long_score / (1.0 + F.softplus(self.kappa))

        ortho = torch.mean((semantic_state * economic_state[:, : semantic_state.size(1)]).sum(dim=-1).pow(2))
        propensity = self.item_propensity[candidate_items].clamp(min=1e-6)
        snips_w = torch.clamp(1.0 / propensity, max=self.tau_propensity_clip)
        snips_w = snips_w / (snips_w.mean(dim=1, keepdim=True) + 1e-8)

        return {
            'total_score': total_score,
            'short_score': short_score,
            'long_score': long_score,
            'do_abs_util': do_abs_util,
            'reference_point': ref,
            'snips_weight': snips_w,
            'ortho_reg': ortho,
        }
