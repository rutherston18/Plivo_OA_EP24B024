"""Trainer — E1: nanochat training-recipe port on the baseline architecture.

Changes vs baseline (architecture in model.py is UNCHANGED for this run):
  * Adam -> AdamW with decoupled weight decay (WD only on 2-D weights)
  * constant LR -> trapezoidal schedule: linear warmup, constant, linear warmdown
  * gradient clipping
  * higher peak LR

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../llm_handout/data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import json
import os
import shutil
import time

import torch

from model import GPT, Config
from muon import Muon
import tokenizer as tokenizer_mod
from evaluate import bits_per_byte

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

# Files snapshotted into the run archive so any checkpoint is fully reproducible.
SNAPSHOT_FILES = ["model.py", "train.py", "muon.py", "tokenizer.py", "evaluate.py", "bpe.json"]


class DummyWandb:
    """No-op stand-in so the training loop is identical with/without --wandb."""
    def log(self, *a, **k): pass
    def finish(self, *a, **k): pass


def archive_run(tag, args, cfg, extra):
    """Copy the exact code + config + checkpoint of this run into experiments/<tag>/."""
    if not tag:
        return
    dst = os.path.join(args.archive_dir, tag)
    os.makedirs(dst, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in SNAPSHOT_FILES:
        src = os.path.join(here, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, fn))
    if os.path.exists(args.out):
        shutil.copy2(args.out, os.path.join(dst, "ckpt.pt"))
    meta = {"args": vars(args),
            "config": {k: getattr(cfg, k) for k in dir(cfg)
                       if not k.startswith("_") and not callable(getattr(cfg, k))}}
    meta.update(extra)
    with open(os.path.join(dst, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"archived run -> {dst}")


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def build_optimizers(model, args):
    """Return a list of optimizers. Each param group carries 'initial_lr' so the
    trapezoidal schedule can scale every group uniformly.

    - optimizer='adamw': everything on AdamW (E1/E2 behavior).
    - optimizer='muon' : 2-D matrices in transformer blocks -> Muon;
                         embeddings + lm_head (+ any 1-D) -> AdamW.
    """
    betas = (args.beta1, args.beta2)
    if args.optimizer == "adamw":
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": args.weight_decay, "initial_lr": args.lr},
            {"params": no_decay, "weight_decay": 0.0, "initial_lr": args.lr},
        ]
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=betas, eps=1e-8)
        return [opt]

    # Muon path: only nn.Linear weights INSIDE blocks (true matmul matrices) go to
    # Muon. Everything else -> AdamW. Note: SSM params like A_log (2-D but not a
    # matmul) and Conv1d weights must NOT go to Muon.
    muon_ids = set()
    muon_params = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod_name.startswith("blocks."):
            muon_params.append(mod.weight)
            muon_ids.add(id(mod.weight))
    adamw_decay, adamw_nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in muon_ids:
            continue
        # weight-decay only on true matrix/lookup weights; skip A_log, Dskip, biases, norms
        if p.dim() >= 2 and "A_log" not in name:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)
    muon = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum,
                weight_decay=args.muon_weight_decay)
    for g in muon.param_groups:
        g["initial_lr"] = args.muon_lr
    adamw_groups = [
        {"params": adamw_decay, "weight_decay": args.weight_decay, "initial_lr": args.lr},
        {"params": adamw_nodecay, "weight_decay": 0.0, "initial_lr": args.lr},
    ]
    adamw = torch.optim.AdamW(adamw_groups, lr=args.lr, betas=betas, eps=1e-8)
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_decay + adamw_nodecay)
    print(f"Muon params: {n_muon:,} | AdamW params: {n_adamw:,}")
    return [muon, adamw]


def lr_multiplier(step, total, warmup, warmdown_ratio, final_frac):
    """Trapezoidal schedule (nanochat-style): linear warmup -> constant ->
    linear warmdown to final_frac * peak."""
    warmdown = round(warmdown_ratio * total)
    if step < warmup:
        return (step + 1) / warmup
    if step <= total - warmdown:
        return 1.0
    progress = (total - step) / max(1, warmdown)  # 1 -> 0 across warmdown
    return progress * 1.0 + (1 - progress) * final_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW LR")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--muon_weight_decay", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--warmdown_ratio", type=float, default=0.4)
    ap.add_argument("--final_lr_frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    # architecture overrides (applied to Config, saved in checkpoint)
    ap.add_argument("--arch", choices=["transformer", "ssm"], default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--tie_weights", action="store_true")
    ap.add_argument("--mlp_act", choices=["relu2", "gelu", "swiglu"], default=None)
    ap.add_argument("--no_mlp", action="store_true", help="SSM: disable per-block MLP")
    ap.add_argument("--mlp_mult", type=int, default=None)
    ap.add_argument("--ssm_state", type=int, default=None)
    ap.add_argument("--ssm_expand", type=int, default=None)
    ap.add_argument("--dt_rank", type=int, default=None)
    ap.add_argument("--hybrid_swa_layer", type=int, default=None)
    # Evaluation during training (for wandb bpb-vs-step charts)
    ap.add_argument("--eval_every", type=int, default=0, help="log dev bpb every N steps (0=off)")
    ap.add_argument("--eval_file", default="../llm_handout/data/dev_eval.txt")
    ap.add_argument("--eval_bytes", type=int, default=20000, help="bytes of dev used for periodic eval (0=full; final eval is always full)")
    # wandb + run archiving
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="plivo-llm-speedrun")
    ap.add_argument("--wandb_mode", default="offline", choices=["online", "offline"],
                    help="offline = instant, no network hang; sync later with `wandb sync`")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--tag", default=None, help="archive run scripts+ckpt to experiments/<tag>/")
    ap.add_argument("--archive_dir", default="../experiments")
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    # cache the BPE-encoded corpus so repeated runs skip the ~22s re-encode
    here = os.path.dirname(os.path.abspath(__file__))
    bpe_path = os.path.join(here, "bpe.json")
    cache = os.path.join(here, f".corpus_ids_v{tok.vocab_size}.pt")
    fresh = (os.path.exists(cache) and os.path.exists(bpe_path)
             and os.path.getmtime(cache) >= os.path.getmtime(bpe_path))
    if fresh:
        ids = torch.load(cache, weights_only=True)
    else:
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        torch.save(ids, cache)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}){' [cached]' if fresh else ''}")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    # apply architecture overrides
    for name in ["arch", "n_layer", "n_embd", "n_head", "block_size", "mlp_act",
                 "mlp_mult", "ssm_state", "ssm_expand", "dt_rank", "hybrid_swa_layer"]:
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.tie_weights:
        cfg.tie_weights = True
    if args.no_mlp:
        cfg.use_mlp = False
    print(f"arch={cfg.arch} n_layer={cfg.n_layer} n_embd={cfg.n_embd} "
          f"tie={cfg.tie_weights} mlp={cfg.mlp_act}x{cfg.mlp_mult}(use={cfg.use_mlp}) "
          f"ssm(N={cfg.ssm_state},exp={cfg.ssm_expand},dt={cfg.dt_rank},swa@{cfg.hybrid_swa_layer})")
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    optimizers = build_optimizers(model, args)

    # wandb (fail-safe: any wandb problem must NOT abort a multi-minute run)
    run = DummyWandb()
    if args.wandb:
        try:
            import wandb
            for k, v in {"WANDB_DIR": "/tmp/wandb", "WANDB_CACHE_DIR": "/tmp/wandb-cache",
                         "WANDB_CONFIG_DIR": "/tmp/wandb-config"}.items():
                os.environ.setdefault(k, v)
                os.makedirs(v, exist_ok=True)
            run = wandb.init(project=args.wandb_project, name=args.run_name,
                             mode=args.wandb_mode,
                             settings=wandb.Settings(init_timeout=30),
                             config={**vars(args), "n_params": n})
        except Exception as e:
            print(f"[wandb disabled] init failed: {e}")
            run = DummyWandb()

    eval_text = None
    if args.eval_every > 0:
        eval_text = open(args.eval_file, encoding="utf-8").read()

    def eval_bpb(full=False):
        text = eval_text if (full or not args.eval_bytes) else eval_text[:args.eval_bytes]
        model.eval()
        with torch.no_grad():
            bpb, _, _ = bits_per_byte(model, cfg, tok, text)
        model.train()
        return bpb

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lrm = lr_multiplier(step - 1, args.steps, args.warmup,
                            args.warmdown_ratio, args.final_lr_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * lrm

        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = 0.0
        if args.grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        for opt in optimizers:
            opt.step()
        losses.append(loss.item())

        log = {"step": step, "train/loss": loss.item(), "lr_mult": lrm,
               "grad_norm": gnorm}
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            log["val/bpb"] = eval_bpb()
            print(f"  [eval] step {step} dev bpb {log['val/bpb']:.4f}")
        run.log(log)

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lrm {lrm:.3f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

    final_bpb = eval_bpb(full=True) if args.eval_every > 0 else None
    if final_bpb is not None:
        run.log({"final/bpb": final_bpb})
    archive_run(args.tag, args, cfg, {"final_train_loss": losses[-1],
                                      "final_dev_bpb": final_bpb,
                                      "n_params": n})
    run.finish()


if __name__ == "__main__":
    main()
