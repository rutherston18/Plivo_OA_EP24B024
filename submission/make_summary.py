"""Generate submission/SUMMARY.html: compiles the whole project (repo scan,
each improvement + why, wandb logging, and a per-experiment appendix) with plots
built from the saved training logs / loss curves.

Usage:
    python make_summary.py                 # build SUMMARY.html
    python make_summary.py --wandb-backfill # also push historical curves to wandb
"""
import argparse
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Per-run metadata (label/description + appendix narrative), keyed by folder name.
# load_runs() auto-discovers any experiments/<dir> with a ckpt.pt + dev_bpb.json,
# so deliverables stay current after each experiment without editing this file.
META = {
    "baseline_results": dict(label="Baseline", order=0, color="#9e9e9e",
        desc="Starter GPT, byte tokenizer, Adam const-LR",
        hyp="", chg="", concl="Unmodified starter code. The reference every experiment is measured against."),
    "e1_train_recipe": dict(label="E1", order=1, color="#4c8bf5",
        desc="Trapezoidal LR + AdamW + grad clip",
        hyp="The baseline is under-trained: constant LR, no warmup/decay, plain Adam.",
        chg="AdamW (wd 0.1 on 2-D weights), trapezoidal LR (100-step warmup, warmdown over last 40% to 0), peak LR 3e-4&rarr;1e-3, grad clip 1.0. Architecture unchanged.",
        concl="Big cheap win from optimization alone; loss falls fastest during warmdown."),
    "e2_modern_arch": dict(label="E2", order=2, color="#f5a623",
        desc="RoPE + RMSNorm + QK-norm + ReLU2 + zero-init + softcap",
        hyp="The starter block (learned pos, LayerNorm, GELU, biases) is dated.",
        chg="RoPE (drop learned pos), parameter-free RMSNorm, QK-norm, ReLU&sup2;, bias=False, zero-init output projections, logit softcap 15.",
        concl="Modernization helps and is 'free' on params (fewer than baseline)."),
    "e3_muon": dict(label="E3", order=3, color="#2ecc71",
        desc="Muon optimizer (Newton-Schulz)",
        hyp="At a fixed 2000-step budget, per-step efficiency matters most.",
        chg="Pure-PyTorch single-device Muon (NS-5, momentum 0.95, lr 0.02) for block matrices; AdamW (lr 1e-3) for embeddings+head.",
        concl="Muon converges faster per step. Under 2.0 BPB for the first time."),
    "e4_bpe_tie": dict(label="E4", order=4, color="#9b59b6",
        desc="Byte-level BPE-1024 tokenizer + weight tying",
        hyp="Byte tokenizer wastes 3-4 tokens per Devanagari char; BPB is per-byte so compression directly lowers the score.",
        chg="Pure-Python lossless BPE-1024 (dev 2.33 bytes/token) + tied input/output embeddings (frees ~164k params). Muon.",
        concl="Largest single win so far; a 128-token context now spans ~300 bytes."),
    "e5_swiglu": dict(label="E5", order=5, color="#e74c3c",
        desc="SwiGLU MLP + depth/width tuning", hyp="", chg="", concl=""),
    "e6_ngram": dict(label="E6", order=6, color="#1abc9c",
        desc="Hash n-gram / Engram embeddings", hyp="", chg="", concl=""),
    "e7_ssm": dict(label="E7", order=7, color="#e67e22",
        desc="Selective SSM backbone (Mamba-1), maxed params", hyp="", chg="", concl=""),
}


def _dirs():
    yield os.path.join(ROOT, "baseline_results")
    exp = os.path.join(ROOT, "experiments")
    if os.path.isdir(exp):
        for name in sorted(os.listdir(exp)):
            yield os.path.join(exp, name)


def load_runs():
    out = []
    for d in _dirs():
        name = os.path.basename(d)
        ckpt_p, bpb_p = os.path.join(d, "ckpt.pt"), os.path.join(d, "dev_bpb.json")
        if name == "smoke_test" or not (os.path.exists(ckpt_p) and os.path.exists(bpb_p)):
            continue
        m = META.get(name, dict(label=name, order=99, color="#7f8c8d", desc=name, hyp="", chg="", concl=""))
        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        bpb = json.load(open(bpb_p))
        out.append(dict(label=m["label"], desc=m["desc"], color=m["color"], order=m["order"],
                        hyp=m["hyp"], chg=m["chg"], concl=m["concl"],
                        curve=ck.get("train_loss_curve", []), steps=ck.get("steps"),
                        bpb=bpb["bpb"], n_params=bpb["n_params"]))
    out.sort(key=lambda r: r["order"])
    return out


def color_of(r):
    return r["color"]


def ema(xs, beta=0.98):
    y, m = [], 0.0
    for i, x in enumerate(xs):
        m = beta * m + (1 - beta) * x
        y.append(m / (1 - beta ** (i + 1)))
    return y


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_loss(runs):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for r in runs:
        if r["curve"]:
            ax.plot(range(1, len(r["curve"]) + 1), ema(r["curve"]),
                    label=f"{r['label']}  (bpb {r['bpb']})", color=r["color"], lw=2)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (EMA, nats/token)")
    ax.set_title("Training loss curves — smoothed"); ax.legend(); ax.grid(alpha=0.25)
    return fig_to_b64(fig)


def plot_bpb(runs):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [r["label"] for r in runs]
    vals = [r["bpb"] for r in runs]
    bars = ax.bar(labels, vals, color=[r["color"] for r in runs], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    base = vals[0]
    for i, v in enumerate(vals):
        if i > 0:
            ax.text(i, v / 2, f"-{(base - v) / base * 100:.1f}%", ha="center", color="white", fontsize=10, fontweight="bold")
    ax.set_ylabel("dev bits-per-byte (lower = better)")
    ax.set_title("Dev BPB by experiment (cumulative improvement vs baseline)")
    ax.set_ylim(0, max(vals) * 1.15); ax.grid(axis="y", alpha=0.25)
    return fig_to_b64(fig)


def wandb_backfill(runs):
    import wandb
    for r in runs:
        run = wandb.init(project="plivo-llm-speedrun", name=f"backfill-{r['label']}",
                         config={"n_params": r["n_params"], "steps": r["steps"], "desc": r["desc"]},
                         reinit=True)
        for i, l in enumerate(r["curve"], 1):
            run.log({"step": i, "train/loss": l})
        run.log({"final/dev_bpb": r["bpb"]})
        run.summary["dev_bpb"] = r["bpb"]
        run.finish()
    print("wandb backfill complete")


# --------------------------------------------------------------------------- #
# HTML content
# --------------------------------------------------------------------------- #

CSS = """
:root{--bg:#0f1220;--card:#1a1f36;--ink:#e8ecf5;--mut:#9aa5c4;--acc:#4c8bf5;--good:#2ecc71;--warn:#f5a623;}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--ink);line-height:1.6}
.wrap{max-width:980px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:30px;margin:.2em 0}h2{margin-top:1.8em;border-bottom:1px solid #2b3358;padding-bottom:.3em}
h3{margin-top:1.4em;color:#cfd8ff}code{background:#11142a;padding:2px 6px;border-radius:5px;font-size:.9em}
.sub{color:var(--mut);font-size:15px}
.card{background:var(--card);border:1px solid #2b3358;border-radius:12px;padding:18px 20px;margin:16px 0}
.hl{font-size:22px;font-weight:700;color:var(--good)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14.5px}
th,td{border:1px solid #2b3358;padding:8px 10px;text-align:left}th{background:#20264a}
tr:nth-child(even){background:#171c33}
.tag{display:inline-block;background:#20264a;border:1px solid #34406e;border-radius:20px;padding:2px 10px;margin:2px;font-size:12.5px;color:#cfd8ff}
img{max-width:100%;border-radius:10px;border:1px solid #2b3358;margin:8px 0;background:#fff}
.good{color:var(--good)}.warn{color:var(--warn)}.mut{color:var(--mut)}
ul{margin:.4em 0}.pill{font-size:12px;padding:1px 8px;border-radius:12px;background:#123;border:1px solid #2b3358}
.kv{display:flex;flex-wrap:wrap;gap:8px}.kv div{background:#11142a;border:1px solid #2b3358;border-radius:8px;padding:6px 10px;font-size:13px}
"""


def build_html(runs, loss_png, bpb_png):
    best = min(runs, key=lambda r: r["bpb"])
    base = runs[0]
    row_list = []
    for r in runs:
        if r["label"] == "Baseline":
            vs = "&mdash;"
        else:
            vs = f"-{(base['bpb'] - r['bpb']) / base['bpb'] * 100:.1f}%"
        row_list.append(
            f"<tr><td><b>{r['label']}</b></td><td>{r['desc']}</td><td>{r['n_params']:,}</td>"
            f"<td>{r['bpb']:.4f}</td><td>{vs}</td></tr>")
    rows = "".join(row_list)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>2,000-Step LLM Speedrun — Summary</title><style>{CSS}</style></head><body><div class="wrap">

<h1>2,000-Step LLM Speedrun &mdash; Project Summary</h1>
<div class="sub">Mixed English+Hindi, CPU-only, &le;2000 steps, &le;2M params, pure PyTorch. Metric: dev <b>bits-per-byte</b> (lower is better).</div>

<div class="card">
  <div>Best result so far: <span class="hl">dev BPB {best['bpb']:.4f}</span> &nbsp;(<b>{best['label']}</b>),
  down from baseline {base['bpb']:.4f} &mdash; a <b class="good">{(base['bpb']-best['bpb'])/base['bpb']*100:.1f}%</b> reduction, at {best['n_params']:,} params.</div>
  <div class="mut" style="margin-top:8px">Experiments E1&ndash;E4 completed and scored below (each snapshotted for reproducibility). E5&ndash;E7 (SwiGLU, hash n-gram, SSM backbone) are queued.</div>
</div>

<h2>1. Results at a glance</h2>
<table><tr><th>Run</th><th>What changed</th><th>Params</th><th>Dev BPB</th><th>vs baseline</th></tr>{rows}</table>
<img src="data:image/png;base64,{bpb_png}" alt="dev bpb by experiment">
<img src="data:image/png;base64,{loss_png}" alt="training loss curves">
<div class="mut">Plots are generated from the per-step training loss curves saved in each checkpoint and the official scorer's dev-BPB JSON. From E4 onward, the same metrics stream live to Weights &amp; Biases (see &sect;5).</div>

<h2>2. The problem &amp; the constraints</h2>
<p>Train a byte-level-scorable language model on ~7&nbsp;MB of mixed English+Hindi under hard caps:
<span class="tag">&le;2000 optimizer steps</span><span class="tag">&le;2,000,000 params</span><span class="tag">CPU only</span>
<span class="tag">pure PyTorch + numpy + stdlib</span><span class="tag">only train_corpus.txt</span>
<span class="tag">no tiktoken / rustbpe / flash-attn / custom kernels</span>.
The scorer runs <code>python evaluate.py --checkpoint ckpt.pt --text_file &lt;file&gt;</code> and the tokenizer must be lossless.</p>
<p>Because steps and params are capped, the game is <b>sample efficiency</b> (get the most out of 2000 steps) and
<b>parameter efficiency</b> (spend the 2M budget where it matters), plus a tokenizer that compresses Devanagari
(3&ndash;4 bytes/char) so a fixed context covers more text and each predicted token covers more bytes.</p>

<h2>3. Repositories scanned &amp; what we took from each</h2>
<div class="card"><h3>nanochat (Karpathy)</h3>
<div class="kv"><div>Muon optimizer</div><div>RoPE</div><div>QK-norm</div><div>RMSNorm</div><div>ReLU&sup2; MLP</div><div>zero-init projections</div><div>logit softcap</div><div>trapezoidal LR</div></div>
<p class="mut">The modern-transformer + optimization recipe. Muon (Newton&ndash;Schulz orthogonalized updates) and the trapezoidal LR schedule are the two biggest sample-efficiency levers for a fixed step budget. Ported to single-device, pure-PyTorch (dropped its distributed / torch.compile / fp8 / FA3 machinery, which the PS forbids or which don't help on CPU).</p></div>

<div class="card"><h3>MeowLLM</h3>
<div class="kv"><div>Tied embeddings</div><div>SwiGLU</div><div>RoPE</div><div>RMSNorm</div><div>depth-scaled init</div></div>
<p class="mut">A compact modern decoder that confirms the PS advice: <b>weight tying is non-negotiable</b> (frees ~164k params at vocab 1024 to reinvest in depth/width) and <b>SwiGLU</b> gives better capacity-per-parameter than a plain 4&times; MLP. Queued as E4/E5.</p></div>

<div class="card"><h3>SiliconLLM</h3>
<div class="kv"><div>Mamba-1 selective SSM (ArchA)</div><div>sliding-window attn hybrid</div><div>dReLU sparse MLP</div><div>ternary BitLinear</div><div>n-gram drafter</div><div>associative recall</div></div>
<p class="mut">A deep CPU-native LLM research repo. Its pure-PyTorch <b>selective SSM backbone</b> (linear-time in sequence length, cache-friendly on CPU) is our blueprint for the State-Space-Model architecture (queued as E7, now the priority direction). Its n-gram asset is an offline decode-time drafter, so the <b>hash n-gram / Engram</b> idea (family&nbsp;2) will be built fresh as a trainable module.</p></div>

<h2>4. Every improvement &mdash; what it does &amp; why it helps</h2>
{improvements_html()}

<h2>5. Weights &amp; Biases logging &amp; reproducibility</h2>
<p><code>train.py</code> integrates wandb (<code>--wandb</code>, project <code>plivo-llm-speedrun</code>). Per step it logs
<code>train/loss</code>, <code>lr_mult</code> and <code>grad_norm</code>; with <code>--eval_every</code> it logs
<code>val/bpb</code> vs step (on a cheap dev subset, with a full dev eval at the end) so the BPB-vs-step curve is
visible live. This makes the effect of each aggressive-LR variant observable without waiting for the run to finish.</p>
<p><b>Reproducibility:</b> every run started with <code>--tag &lt;name&gt;</code> snapshots the exact
<code>model.py, train.py, muon.py, tokenizer.py, evaluate.py, bpe.json</code> + <code>ckpt.pt</code> +
<code>run_meta.json</code> (args &amp; config) into <code>experiments/&lt;name&gt;/</code>, so any checkpoint can be rebuilt or resumed.</p>

<h2>6. Machine vs. human contribution</h2>
<p class="mut">This project was built with an AI coding assistant (Cursor) under continuous human direction. The
<b>human</b> set the strategy and priorities (which repos/ideas to port, "build an SSM and max out params", "train
more aggressively", the ask-before-full-runs and archive-everything workflow, and all go/no-go decisions). The
<b>assistant</b> did the implementation: reading the repos, porting Muon/RoPE/SSM/BPE into pure PyTorch under the
caps, wiring wandb + archiving, running the experiments, and writing this report. Every architectural change was
validated by the official scorer, and each run's hypothesis&rarr;result&rarr;conclusion is logged in
<code>RUNLOG.md</code> (reproduced in the appendix).</p>

<h2>Appendix &mdash; experiments in detail</h2>
{appendix_html(runs, base)}

<div class="mut" style="margin-top:40px">Generated by <code>make_summary.py</code> from checkpoints + scorer output. Best dev BPB to date: <b class="good">{best['bpb']:.4f}</b> ({best['label']}).</div>
</div></body></html>"""


def improvements_html():
    items = [
        ("Trapezoidal LR schedule (warmup&rarr;const&rarr;linear warmdown)", "training",
         "Replaces the baseline's constant LR. With only 2000 steps, warmup lets you use a high peak LR safely and the warmdown phase is where loss drops fastest.", "E1"),
        ("AdamW + decoupled weight decay + gradient clipping", "training",
         "Decoupled WD regularizes the 2-D weights; clipping stabilizes the high-LR training the step budget demands.", "E1"),
        ("RoPE (rotary positions)", "architecture",
         "Relative positions injected in attention; removes the learned position table (frees params) and generalizes better across positions.", "E2"),
        ("Parameter-free RMSNorm", "architecture",
         "Cheaper and more stable than LayerNorm; no learnable affine params to spend budget on.", "E2"),
        ("QK-norm", "architecture", "Normalizes queries/keys before attention, preventing logit blow-ups at high LR.", "E2"),
        ("ReLU&sup2; MLP + zero-init output projections + logit softcap", "architecture",
         "ReLU&sup2; is a strong cheap activation; zero-init projections make each residual block an identity at init (clean start &asymp; ln(V) loss); softcap keeps early logits bounded.", "E2"),
        ("Muon optimizer (Newton&ndash;Schulz orthogonalized updates)", "optimizer",
         "Orthogonalizes each 2-D gradient update, which converges faster per step than Adam &mdash; exactly what a fixed 2000-step budget rewards. Embeddings/head stay on AdamW.", "E3"),
        ("Byte-level BPE-1024 tokenizer (pure Python, lossless) + weight tying", "tokenizer",
         "Compresses multi-byte Devanagari into single tokens (dev: 2.33 bytes/token, Hindi 4.0). Since BPB is per-byte, each token covering more bytes directly lowers the score, and a fixed 128-token context now spans ~300 bytes. Tying input/output embeddings frees ~164k params to reinvest.", "E4 (done)"),
        ("SwiGLU MLP", "architecture",
         "SwiGLU gives better capacity per parameter than a plain 4&times; MLP.", "E5 (planned)"),
        ("Hash n-gram / Engram embeddings", "family 2",
         "Cheap hashed bigram/trigram lookups injected into the stream give tiny models instant local spelling/morphology memory, freeing attention/depth for higher-level structure.", "E6 (planned)"),
        ("Selective State-Space Model backbone (Mamba-1)", "family 3",
         "Linear-time O(N) recurrence instead of O(N&sup2;) attention &mdash; cache-friendly on memory-bandwidth-bound CPUs; lets us boost hidden dim and max out the 2M param budget.", "E7 (priority)"),
        ("Hierarchical / dynamic patching (BLT / SpaceByte)", "family 1",
         "Group bytes into patches so heavy layers see shorter sequences. Deprioritized: BPE already captures most of this benefit, losslessly.", "E8 (stretch)"),
    ]
    cards = []
    for name, cat, why, when in items:
        cards.append(f'<div class="card"><h3>{name} <span class="pill">{cat}</span> <span class="pill">{when}</span></h3><p class="mut">{why}</p></div>')
    return "\n".join(cards)


def appendix_html(runs, base):
    out = []
    done_labels = set()
    for r in runs:
        done_labels.add(r["label"])
        if r["label"] == "Baseline":
            out.append(f'<div class="card"><h3>Baseline &mdash; dev BPB {r["bpb"]:.4f} ({r["n_params"]:,} params)</h3>'
                       f'<p class="mut">{r["concl"]}</p></div>')
            continue
        delta = f'-{(base["bpb"]-r["bpb"])/base["bpb"]*100:.1f}% vs baseline'
        out.append(f'<div class="card"><h3>{r["label"]} &mdash; dev BPB {r["bpb"]:.4f} <span class="pill">{delta}</span></h3>'
                   f'<p><b>Hypothesis.</b> {r["hyp"]}</p><p><b>Changed.</b> {r["chg"]}</p>'
                   f'<p><b>Result.</b> dev BPB {r["bpb"]:.4f} ({r["n_params"]:,} params, {r["steps"]} steps).</p>'
                   f'<p><b>Conclusion.</b> {r["concl"]}</p></div>')
    # queued = META entries not yet completed
    queued = [m for k, m in sorted(META.items(), key=lambda kv: kv[1]["order"])
              if m["label"] not in done_labels and m["order"] > 0]
    if queued:
        items = "".join(f'<li><b>{m["label"]}</b> {m["desc"]}</li>' for m in queued)
        out.append(f'<div class="card"><h3>Queued</h3><ul>{items}</ul></div>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-backfill", action="store_true")
    args = ap.parse_args()
    runs = load_runs()
    assert runs, "no runs found"
    loss_png = plot_loss(runs)
    bpb_png = plot_bpb(runs)
    html = build_html(runs, loss_png, bpb_png)
    out_p = os.path.join(os.path.dirname(__file__), "SUMMARY.html")
    with open(out_p, "w") as f:
        f.write(html)
    print(f"wrote {out_p} ({len(html)//1024} KB) from {len(runs)} runs")
    if args.wandb_backfill:
        wandb_backfill(runs)


if __name__ == "__main__":
    main()
