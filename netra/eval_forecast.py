"""Does the Health Passport forecast the day a camera will cross a fault limit?

Ground truth = the first day the diagnosis itself flags the fault (2 days in a row) on the same
timeline continued to day 99. Forecasts only see the history up to the checkpoint day.
"""
from __future__ import annotations
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from . import scenes, timeline
from .diagnose import diagnose
from .passport import Passport, WATCHES

DAYS = 100
CHECKPOINTS = (20, 30)
STABLE_CHECKS = (20, 30, 40, 60)


def run_timeline(args):
    scene, process, seed = args
    cam = Passport(f"{scene}/{process}")
    flagged = []
    for d, ref, fr in timeline.frames(scene, process, DAYS, seed):
        diag = diagnose(fr, ref)
        cam.record(d, diag.metrics, diag.faults())
        flagged.append(diag.faults())
    target = timeline.FAULT_OF.get(process)
    actual = None
    if target:
        for d in range(DAYS - 1):
            if target in flagged[d] and target in flagged[d + 1]:
                actual = d
                break
    out = {"scene": scene, "process": process, "target": target, "actual_day": actual, "checks": []}
    cps = CHECKPOINTS if target else STABLE_CHECKS
    for cp in cps:
        alerts = cam.report(now=float(cp), horizon_days=30.0)
        out["checks"].append({"checkpoint": cp, "alerts": [
            {"fault": a["fault"], "eta_days": a["eta_days"], "p": a["p"]} for a in alerts]})
    return out


def summarise(rows):
    S = {"per_process": {}}
    errs_all, lead_all = [], []
    for proc in timeline.PROCESSES[:-1]:
        rs = [r for r in rows if r["process"] == proc]
        tgt = timeline.FAULT_OF[proc]
        per = {}
        for cp in CHECKPOINTS:
            hits, errs, extra = 0, [], 0
            for r in rs:
                if r["actual_day"] is None or r["actual_day"] <= cp:
                    continue
                al = [a for c in r["checks"] if c["checkpoint"] == cp for a in c["alerts"]]
                mine = [a for a in al if a["fault"] == tgt]
                extra += len(al) - len(mine)
                if mine:
                    hits += 1
                    errs.append(abs(cp + mine[0]["eta_days"] - r["actual_day"]))
            n = len([r for r in rs if r["actual_day"] is not None and r["actual_day"] > cp])
            per[f"day{cp}"] = {"n": n, "alerted": hits,
                               "mae_days": round(float(np.mean(errs)), 1) if errs else None,
                               "median_abs_err_days": round(float(np.median(errs)), 1) if errs else None,
                               "wrong_fault_alerts": extra}
            errs_all += errs
        per["actual_flag_days"] = [r["actual_day"] for r in rs]
        S["per_process"][proc] = per
    st = [r for r in rows if r["process"] == "stable"]
    checks = [c for r in st for c in r["checks"]]
    S["stable_cameras"] = {"timelines": len(st), "checks": len(checks),
                           "false_alerts": sum(len(c["alerts"]) for c in checks)}
    S["overall_mae_days"] = round(float(np.mean(errs_all)), 1) if errs_all else None
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/forecast")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--scenes", nargs="*", default=scenes.FRESH_SCENES[:6])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, p, 100 + i) for i, sc in enumerate(a.scenes) for p in timeline.PROCESSES]
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(run_timeline, jobs)
    S = summarise(rows)
    S["meta"] = {"scenes": a.scenes, "days": DAYS, "wall_seconds": round(time.time() - t0, 1)}
    (out / "forecast_summary.json").write_text(json.dumps(S, indent=2))
    (out / "forecast_runs.json").write_text(json.dumps(rows))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
