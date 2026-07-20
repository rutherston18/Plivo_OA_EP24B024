"""Muon optimizer — single-device, pure-PyTorch port from nanochat/modded-nanogpt.

Muon = SGD-momentum whose 2-D update is orthogonalized by a Newton-Schulz
iteration (the "zeropower" step), which empirically converges much faster than
Adam per optimizer step. That property is exactly what a fixed 2,000-step budget
rewards. Use Muon only for 2-D matrix weights; embeddings, the final head and any
{0,1}-D params should stay on AdamW.

Stripped of everything the PS forbids/doesn't need: no distributed comm, no
torch.compile fused kernels, no fp8/bf16 tricks. Just matmuls in fp32 on CPU.
Ref: https://kellerjordan.github.io/posts/muon/
"""
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Compute an approximate orthogonalization U V^T of G via a quintic
    Newton-Schulz iteration. Coefficients from modded-nanogpt (tuned to maximize
    slope at zero). Runs in fp32 for CPU stability."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + eps)  # ensure spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon is for 2-D params only"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so the update RMS matches SGD across non-square shapes
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g.to(p.dtype), alpha=-lr * scale)
