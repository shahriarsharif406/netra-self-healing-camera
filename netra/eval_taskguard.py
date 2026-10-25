"""Does a downstream-task guard help? Same frames, same agent, with and without `task=` (QR decoding in the
user's fixed regions). The guard sees only the candidate frame's task score, never ground truth images.
Caveat: the task score uses the expected payloads, i.e. the deployer must have a way to score the task."""
from __future__ import annotations
import argparse, json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from . import metrics, scenes
from .agent import Agent
from .faults import FAULTS, apply_faults

SEV = {"low": 0.25, "mid": 0.6, "high": 0.95}


def one(args):
    name, idx = args
    rng = np.random.default_rng(5000 + idx)
    ref, live, pl = scenes.make_pair(name, rng)
    roi = lambda im: metrics.qr_decoded_roi(im, pl, scenes.QR_POSITIONS, scenes.QR_SIZE) / len(pl)
    plain = Agent(ref, policy="auto_all")
    guard = Agent(ref, policy="auto_all", task=roi)
    rows = []
    for f in FAULTS:
        for sn, sv in SEV.items():
            bad, _ = apply_faults(live, [f], rng, sev=sv)
            a, b = plain.run(bad), guard.run(bad)
            rows.append(dict(scene=name, fault=f, sev=sn, task_bad=roi(bad), task_plain=roi(a.output),
                             task_guard=roi(b.output),
                             ssim_plain=metrics.ssim(a.output, live), ssim_guard=metrics.ssim(b.output, live),
                             guard_fired=sum(1 for s in b.steps if s.get("task_guard"))))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/taskguard")
    ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as p:
        rows = [r for rs in p.map(one, list(zip(scenes.FRESH_SCENES, range(8)))) for r in rs]
    m = lambda k, sel=lambda r: True: round(float(np.mean([r_[k] for r_ in rows if sel(r_)])), 3)
    S = {"n": len(rows), "task_bad": m("task_bad"), "task_plain": m("task_plain"), "task_guard": m("task_guard"),
         "ssim_plain": m("ssim_plain"), "ssim_guard": m("ssim_guard"),
         "frames_guard_fired": sum(r["guard_fired"] > 0 for r in rows),
         "task_worse_than_bad_plain": sum(r["task_plain"] < r["task_bad"] - 1e-9 for r in rows),
         "task_worse_than_bad_guard": sum(r["task_guard"] < r["task_bad"] - 1e-9 for r in rows),
         "by_fault": {f: {k: m(k, lambda r, f=f: r["fault"] == f) for k in
                          ("task_bad", "task_plain", "task_guard", "ssim_plain", "ssim_guard")} for f in FAULTS}}
    (out / "summary.json").write_text(json.dumps(S, indent=2)); (out / "runs.json").write_text(json.dumps(rows))
    print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
