"""Figures for the README / demo video:  python -m netra.report results/final_fresh"""
import sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics, scenes
from .agent import Agent
from .faults import FAULTS, apply_faults

C = {"bad": "#b0b7c3", "naive": "#e08a4b", "agent": "#2a7f62"}


def fig_ssim(d: pd.DataFrame, out: Path):
    groups = [(f, d[(d.kind == "single") & (d.truth == f)]) for f in FAULTS]
    groups += [("compound", d[d.kind == "compound"]), ("clean", d[d.kind == "clean"])]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (k, lab) in enumerate((("bad", "faulty frame (no action)"), ("naive", "naive 'always enhance'"),
                                  ("agent", "NETRA agent"))):
        ax.bar(x + (i - 1) * 0.27, [g[f"ssim_{k}"].mean() for _, g in groups], 0.27, label=lab, color=C[k])
    ax.set_xticks(x, [g[0].replace("_", " ") for g in groups])
    ax.set_ylabel("mean SSIM vs ground truth (higher is better)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_ssim_by_fault.png", dpi=160)
    plt.close(fig)


def fig_confusion(d: pd.DataFrame, out: Path):
    s = d[d.kind.isin(["single", "clean"])]
    rows = ["clean"] + FAULTS
    cols = ["clean"] + FAULTS + ["multiple"]
    M = np.zeros((len(rows), len(cols)))
    for _, r in s.iterrows():
        p = r.pred
        j = cols.index(p) if p in cols else cols.index("multiple")
        M[rows.index(r.truth), j] += 1
    M = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("first diagnosis")
    ax.set_ylabel("injected fault")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.png", dpi=160)
    plt.close(fig)


def gallery(out: Path, scene="collage2", seed=5):
    rng = np.random.default_rng(seed)
    ref, live, _ = scenes.make_pair(scene, rng)
    agent = Agent(ref, policy="auto_all")
    cases = [[f] for f in FAULTS] + [["drift", "noise"], ["low_light", "blur"]]
    W, H = 300, 225
    rows = []
    for c in cases:
        bad, _ = apply_faults(live, c, rng, sev=0.75)
        res = agent.run(bad)
        tiles = []
        for img, lab in ((bad, f"{'+'.join(c)}  SSIM {metrics.ssim(live, bad):.2f}"),
                         (res.output, f"NETRA  SSIM {metrics.ssim(live, res.output):.2f}"),
                         (live, "ground truth")):
            t = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).copy()
            cv2.rectangle(t, (0, 0), (W, 20), (0, 0, 0), -1)
            cv2.putText(t, lab, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "gallery.png"), np.vstack(rows))


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = src / "figures"
    out.mkdir(exist_ok=True)
    d = pd.read_csv(src / "eval_runs.csv")
    fig_ssim(d, out)
    fig_confusion(d, out)
    gallery(out)
    print("wrote", sorted(p.name for p in out.iterdir()))
