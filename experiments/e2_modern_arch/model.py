"""GPT — E2: nanochat-style architecture modernization, plain PyTorch.

vs the starter model:
  * RoPE (rotary position embeddings) instead of a learned positional table
  * RMSNorm with no learnable params instead of LayerNorm
  * QK-norm on queries/keys before attention
  * ReLU^2 MLP activation instead of GELU
  * no bias in linear layers
  * zero-initialized output projections (attn.proj, mlp down-proj)
  * logit softcap: logits = cap * tanh(logits / cap)

Every knob is a Config flag so the checkpoint records it and evaluate.py can
rebuild the exact same model. Parameter cap (2M) and the evaluate.py interface
(forward(idx, targets=None) -> (logits, loss)) are preserved.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = False
    # E2 architecture switches
    use_rope = True
    rope_base = 10000.0
    qk_norm = True
    rmsnorm = True
    bias = False
    mlp_act = "relu2"        # "relu2" | "gelu"
    logit_softcap = 15.0     # 0 disables
    zero_init_proj = True


def norm(x, use_rmsnorm, ln):
    if use_rmsnorm:
        return F.rms_norm(x, (x.size(-1),))
    return ln(x)


def build_norm(cfg):
    """LayerNorm module only needed when not using parameter-free RMSNorm."""
    return None if cfg.rmsnorm else nn.LayerNorm(cfg.n_embd)


def precompute_rope(head_dim, block_size, base, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                            device=device) / head_dim))
    t = torch.arange(block_size, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    cos, sin = freqs.cos(), freqs.sin()
    return cos, sin                                 # (T, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim)
    T = x.size(-2)
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.cfg.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.cfg.qk_norm:
            q, k = F.rms_norm(q, (hd,)), F.rms_norm(k, (hd,))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() if self.cfg.mlp_act == "relu2" else F.gelu(x)
        return self.drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = build_norm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = build_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x, self.cfg.rmsnorm, self.ln1), cos, sin)
        x = x + self.mlp(norm(x, self.cfg.rmsnorm, self.ln2))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = build_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        if cfg.use_rope:
            hd = cfg.n_embd // cfg.n_head
            cos, sin = precompute_rope(hd, cfg.block_size, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        if cfg.zero_init_proj:
            for blk in self.blocks:
                nn.init.zeros_(blk.attn.proj.weight)
                nn.init.zeros_(blk.mlp.proj.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if not self.cfg.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        cos = self.rope_cos if self.cfg.use_rope else None
        sin = self.rope_sin if self.cfg.use_rope else None
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(norm(x, self.cfg.rmsnorm, self.ln_f))
        cap = self.cfg.logit_softcap
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # parameters() de-duplicates shared tensors, so tied weights count once
        return sum(p.numel() for p in self.parameters())
