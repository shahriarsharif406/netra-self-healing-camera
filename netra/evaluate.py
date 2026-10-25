"""Reproducible evaluation. Thresholds and tool parameters were tuned on CALIB_SCENES only;
everything reported here uses TEST_SCENES that the tuning never saw.

For every frame we know the ground truth (clean live frame), which the agent never sees.
Compared: no-op, a naive "always enhance" pipeline (no diagnosis), and the NETRA agent.
"""
from __future__ import annotations
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
import cv2
import numpy as np

from . import metrics, scenes
from .agent import Agent
from .faults import FAULTS, apply_faults

SEVERITIES = {"low": 0.25, "mid": 0.6, "high": 0.95}
COMPOUND = [["drift", "noise"], ["low_light", "blur"], ["overexposure", "noise"],
            ["drift", "lens_dirt"], ["blur", "noise"], ["drift", "low_light"]]


def naive_enhance(img: np.ndarray) -> np.ndarray:
    """What people do without a diagnosis: CLAHE + denoise + sharpen, on every frame."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)
    return cv2.addWeighted(out, 1.8, cv2.GaussianBlur(out, (0, 0), 1.5), -0.8, 0)


def _qr(img, pl):
    return (metrics.qr_decoded(img, pl),
            metrics.qr_decoded_roi(img, pl, scenes.QR_POSITIONS, scenes.QR_SIZE))


def _row(kind, truth, sev_name, scene, live, bad, out_agent, res, pl, naive):
    q = {"clean": _qr(live, pl), "bad": _qr(bad, pl), "agent": _qr(out_agent, pl), "naive": _qr(naive, pl)}
    r = {"scene": scene, "kind": kind, "truth": "+".join(truth) or "clean", "severity": sev_name,
         "pred": "+".join(res.initial.faults()) or "clean",
         "final_pred": "+".join(res.final.faults()) or "clean",
         # every fault the agent identified at any point of the run (dirt is judged last by design)
         "handled": "+".join(sorted(set(res.initial.faults()) | {s["fault"] for s in res.steps}
                                    | set(res.unresolved))) or "clean",
         "unresolved": "+".join(res.unresolved), "runtime_ms": round(res.runtime_ms, 1),
         "n_steps_applied": sum(s["status"] in ("applied", "partial") for s in res.steps),
         "n_rollbacks": sum(s["status"] == "rolled_back" for s in res.steps)}
    for name, img in (("bad", bad), ("agent", out_agent), ("naive", naive)):
        r[f"ssim_{name}"] = round(metrics.ssim(live, img), 4)
        r[f"psnr_{name}"] = round(metrics.psnr(live, img), 2)
    for name, (g, roi) in q.items():
        r[f"qr_global_{name}"], r[f"qr_roi_{name}"] = g, roi
    return r


def eval_scene(args):
    name, idx = args
    rng = np.random.default_rng(1000 + idx)
    ref, live, pl = scenes.make_pair(name, rng)
    agent = Agent(ref, policy="auto_all")      # auto_all: evaluation has no human in the loop
    rows = []
    cases = [("clean", [], "none")] * 2
    cases += [("single", [f], sn) for f in FAULTS for sn in SEVERITIES]
    cases += [("compound", c, "rand") for c in COMPOUND]
    for kind, truth, sev_name in cases:
        if kind == "clean":
            bad = live
        elif kind == "single":
            bad, _ = apply_faults(live, truth, rng, sev=SEVERITIES[sev_name])
        else:
            bad, _ = apply_faults(live, truth, rng)
        res = agent.run(bad)
        rows.append(_row(kind, truth, sev_name, name, live, bad, res.output, res, pl, naive_enhance(bad)))
    return rows


def summarise(rows: list[dict]) -> dict:
    import collections
    S: dict = {"n_runs": len(rows)}
    # ---- diagnosis quality
    singles = [r for r in rows if r["kind"] == "single"]
    clean = [r for r in rows if r["kind"] == "clean"]
    S["diagnosis"] = {
        "clean_false_alarm_rate": round(float(np.mean([r["pred"] != "clean" for r in clean])), 4),
        "single_exact_match": round(float(np.mean([r["pred"] == r["truth"] for r in singles])), 4),
        "single_target_recall": round(float(np.mean([r["truth"] in r["pred"].split("+") for r in singles])), 4),
    }
    per = {}
    for f in FAULTS:
        tp = sum(f in r["truth"].split("+") and f in r["pred"].split("+") for r in rows)
        fp = sum(f not in r["truth"].split("+") and f in r["pred"].split("+") for r in rows)
        fn = sum(f in r["truth"].split("+") and f not in r["pred"].split("+") for r in rows)
        p_, r_ = tp / max(1, tp + fp), tp / max(1, tp + fn)
        per[f] = {"precision": round(p_, 3), "recall": round(r_, 3),
                  "f1": round(2 * p_ * r_ / max(1e-9, p_ + r_), 3), "tp": tp, "fp": fp, "fn": fn}
    S["diagnosis"]["per_fault"] = per
    cov = {}
    for f in FAULTS:
        tp = sum(f in r["truth"].split("+") and f in r["handled"].split("+") for r in rows)
        fp = sum(f not in r["truth"].split("+") and f in r["handled"].split("+") for r in rows)
        fn = sum(f in r["truth"].split("+") and f not in r["handled"].split("+") for r in rows)
        cov[f] = {"precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
                  "tp": tp, "fp": fp, "fn": fn}
    S["diagnosis"]["run_level_coverage"] = cov
    # ---- recovery
    def block(sel):
        d = {}
        for m in ("ssim", "psnr"):
            for k in ("bad", "naive", "agent"):
                d[f"{m}_{k}"] = round(float(np.mean([r[f"{m}_{k}"] for r in sel])), 3)
        d["agent_improved"] = round(float(np.mean([r["ssim_agent"] > r["ssim_bad"] + 0.01 for r in sel])), 3)
        d["agent_harmed"] = round(float(np.mean([r["ssim_agent"] < r["ssim_bad"] - 0.01 for r in sel])), 3)
        d["naive_improved"] = round(float(np.mean([r["ssim_naive"] > r["ssim_bad"] + 0.01 for r in sel])), 3)
        d["naive_harmed"] = round(float(np.mean([r["ssim_naive"] < r["ssim_bad"] - 0.01 for r in sel])), 3)
        for t in ("qr_global", "qr_roi"):
            for k in ("clean", "bad", "naive", "agent"):
                d[f"{t}_{k}"] = round(float(np.mean([r[f"{t}_{k}"] for r in sel])) / 3, 3)
        d["runtime_ms"] = round(float(np.mean([r["runtime_ms"] for r in sel])), 0)
        d["n"] = len(sel)
        return d
    S["recovery_by_fault"] = {f: block([r for r in singles if r["truth"] == f]) for f in FAULTS}
    S["recovery_by_severity"] = {s: block([r for r in singles if r["severity"] == s]) for s in SEVERITIES}
    S["recovery_compound"] = block([r for r in rows if r["kind"] == "compound"])
    S["recovery_clean"] = block(clean)
    S["recovery_all_faulty"] = block([r for r in rows if r["kind"] != "clean"])
    S["escalated_to_human_rate"] = round(float(np.mean([bool(r["unresolved"]) for r in rows if r["kind"] != "clean"])), 3)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scenes", nargs="*", default=scenes.FRESH_SCENES,
                    help="default: the fresh collage scenes used for the final numbers")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True, parents=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        chunks = pool.map(eval_scene, list(zip(a.scenes, range(len(a.scenes)))))
    rows = [r for c in chunks for r in c]
    import csv
    with open(out / "eval_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    summary["meta"] = {"opencv": cv2.__version__, "test_scenes": a.scenes, "wall_seconds": round(time.time() - t0, 1)}
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["diagnosis"], indent=1))
    print("all-faulty:", json.dumps(summary["recovery_all_faulty"]))
    print(f"done in {time.time() - t0:.0f}s, {len(rows)} runs")


if __name__ == "__main__":
    main()
