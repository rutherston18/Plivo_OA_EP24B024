"""GPT / SSM model — plain PyTorch, selectable architecture.

arch = "transformer" : RoPE + QK-norm attention blocks (E2/E3 lineage)
arch = "ssm"         : Mamba-1 selective diagonal SSM blocks (family 3),
                       optionally with ONE attention layer (hybrid) at
                       hybrid_swa_layer, and an optional per-block MLP.

Shared modern components (nanochat / MeowLLM / SiliconLLM):
  * parameter-free RMSNorm (F.rms_norm)
  * no bias in linear layers
  * weight tying (tok_emb <-> lm_head)  [PS: "non-negotiable"]
  * SwiGLU or ReLU^2 MLP
  * zero-init output projections, logit softcap

Every knob is a Config attribute so it is saved in the checkpoint and
evaluate.py rebuilds the identical model. Caps (2M params) and the
forward(idx, targets=None) -> (logits, loss) interface are preserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # tokenizer / shape
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    # shared switches
    arch = "transformer"        # "transformer" | "ssm"
    tie_weights = False
    bias = False
    rmsnorm = True
    mlp_act = "relu2"           # "relu2" | "gelu" | "swiglu"
    use_mlp = True              # SSM blocks: whether to add an MLP sublayer
    mlp_mult = 4
    logit_softcap = 15.0
    zero_init_proj = True
    # attention (transformer / hybrid attention layer)
    use_rope = True
    rope_base = 10000.0
    qk_norm = True
    # ssm
    ssm_state = 64             # N
    ssm_expand = 2             # d_inner = expand * n_embd
    dt_rank = 16
    ssm_conv = 4
    hybrid_swa_layer = -1      # index of the single attention layer in an SSM stack (-1 = none)
    # hashed n-gram embeddings (family 2): causal token n-gram -> hash bucket -> emb
    ngram_orders = ()          # e.g. (2, 3); empty = disabled
    ngram_buckets = 0          # rows per order's hash table


def rmsnorm(x):
    return F.rms_norm(x, (x.size(-1),))


class Norm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rms = cfg.rmsnorm
        self.ln = None if cfg.rmsnorm else nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        return rmsnorm(x) if self.rms else self.ln(x)


# --------------------------------------------------------------------------- #
# Attention (RoPE + QK-norm)
# --------------------------------------------------------------------------- #

def precompute_rope(head_dim, block_size, base):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(block_size, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    T = x.size(-2)
    cos = cos[:T][None, None]
    sin = sin[:T][None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

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
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


# --------------------------------------------------------------------------- #
# Mamba-1 selective diagonal SSM (ported from SiliconLLM ArchA, pure PyTorch)
# --------------------------------------------------------------------------- #

class SSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.n_embd
        self.D = D
        self.Dn = D * cfg.ssm_expand
        self.N = cfg.ssm_state
        self.dt_rank = cfg.dt_rank
        self.in_proj = nn.Linear(D, 2 * self.Dn, bias=False)
        self.conv1d = nn.Conv1d(self.Dn, self.Dn, cfg.ssm_conv,
                                groups=self.Dn, padding=cfg.ssm_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.Dn, self.dt_rank + 2 * self.N, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.Dn, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.N + 1, dtype=torch.float32).repeat(self.Dn, 1)))
        self.Dskip = nn.Parameter(torch.ones(self.Dn))
        self.out_proj = nn.Linear(self.Dn, D, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xx, z = xz.chunk(2, dim=-1)
        xx = self.conv1d(xx.transpose(1, 2))[:, :, :L].transpose(1, 2)
        xx = F.silu(xx)
        dbl = self.x_proj(xx)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                    # (B,L,Dn)
        A = -torch.exp(self.A_log.float())                   # (Dn,N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                 # (B,L,Dn,N)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xx.unsqueeze(-1)
        h = torch.zeros(B, self.Dn, self.N, device=x.device, dtype=x.dtype)
        ys = []
        dA_t, dBx_t, C_t = dA.unbind(1), dBx.unbind(1), Cm.unbind(1)
        for t in range(L):
            h = dA_t[t] * h + dBx_t[t]
            ys.append((h * C_t[t].unsqueeze(1)).sum(-1))     # (B,Dn)
        y = torch.stack(ys, 1) + xx * self.Dskip
        y = y * F.silu(z)
        return self.out_proj(y)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.act = cfg.mlp_act
        hid = cfg.mlp_mult * cfg.n_embd
        if self.act == "swiglu":
            self.gate = nn.Linear(cfg.n_embd, hid, bias=cfg.bias)
            self.up = nn.Linear(cfg.n_embd, hid, bias=cfg.bias)
            self.down = nn.Linear(hid, cfg.n_embd, bias=cfg.bias)
        else:
            self.fc = nn.Linear(cfg.n_embd, hid, bias=cfg.bias)
            self.proj = nn.Linear(hid, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        if self.act == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        x = self.fc(x)
        x = F.relu(x).square() if self.act == "relu2" else F.gelu(x)
        return self.proj(x)


class NgramEmbed(nn.Module):
    """Causal hashed n-gram embeddings. For each order n and position t, hash the
    tokens (t-n+1 .. t) into `buckets` and look up a learned vector, summed into the
    token embedding. Only uses *input* tokens (<= t), so it's causal / leak-free.
    Cheap local spelling/morphology memory that frees attention+depth for structure."""
    def __init__(self, cfg):
        super().__init__()
        self.orders = tuple(cfg.ngram_orders)
        self.buckets = int(cfg.ngram_buckets)
        self.tables = nn.ModuleDict(
            {str(n): nn.Embedding(self.buckets, cfg.n_embd) for n in self.orders})

    def forward(self, idx):
        out = 0
        for n in self.orders:
            h = torch.zeros_like(idx)
            for k in range(n):
                if k == 0:
                    tok = idx
                else:
                    tok = torch.cat([torch.zeros_like(idx[:, :k]), idx[:, :-k]], dim=1)
                h = (h * 1000003 + tok + 1) % self.buckets
            out = out + self.tables[str(n)](h)
        return out


class Block(nn.Module):
    def __init__(self, cfg, is_attn):
        super().__init__()
        self.cfg = cfg
        self.is_attn = is_attn
        self.norm1 = Norm(cfg)
        self.mix = Attention(cfg) if is_attn else SSM(cfg)
        self.use_mlp = cfg.use_mlp or cfg.arch == "transformer"
        if self.use_mlp:
            self.norm2 = Norm(cfg)
            self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + (self.mix(self.norm1(x), cos, sin) if self.is_attn else self.mix(self.norm1(x)))
        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None
        if cfg.arch == "transformer" and not cfg.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.ngram = (NgramEmbed(cfg)
                      if cfg.ngram_orders and cfg.ngram_buckets > 0 else None)
        self.drop = nn.Dropout(cfg.dropout)

        def layer_is_attn(i):
            if cfg.arch == "transformer":
                return True
            return i == cfg.hybrid_swa_layer     # ssm arch: only this layer is attention
        self.blocks = nn.ModuleList(Block(cfg, layer_is_attn(i)) for i in range(cfg.n_layer))

        self.ln_f = Norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.use_rope = cfg.use_rope and any(b.is_attn for b in self.blocks)
        if self.use_rope:
            hd = cfg.n_embd // cfg.n_head
            cos, sin = precompute_rope(hd, cfg.block_size, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        if cfg.zero_init_proj:
            for b in self.blocks:
                if b.is_attn:
                    nn.init.zeros_(b.mix.proj.weight)
                else:
                    nn.init.zeros_(b.mix.out_proj.weight)
                if b.use_mlp:
                    nn.init.zeros_(b.mlp.down.weight if cfg.mlp_act == "swiglu" else b.mlp.proj.weight)
        if self.ngram is not None:
            for t in self.ngram.tables.values():
                nn.init.zeros_(t.weight)

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
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))[None]
        if self.ngram is not None:
            x = x + self.ngram(idx)
        x = self.drop(x)
        cos = self.rope_cos if self.use_rope else None
        sin = self.rope_sin if self.use_rope else None
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.ln_f(x))
        cap = self.cfg.logit_softcap
        if cap and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
