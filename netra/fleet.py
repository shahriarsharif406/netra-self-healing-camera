"""Fleet view: many cameras, one maintenance queue.

Each camera has a Health Passport (daily diagnoses). The fleet view ranks cameras by urgency:
already past a fault limit first, then by forecast days to the limit, then by health score.
The fleet here is SIMULATED (slow degradations on bundled scenes); the ranking logic is what is real.

Build the cache once:  python -m netra.fleet --build   (about 2-3 minutes)
"""
from __future__ import annotations
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from . import timeline
from .diagnose import diagnose
from .passport import Passport, WATCHES

CACHE = Path(__file__).resolve().parent.parent / "results" / "fleet.json"
DAYS = 50
FLEET = [  # camera id, scene, simulated slow process, seed
    ("cam-lobby", "collage0", "creeping_drift", 11),
    ("cam-loading-dock", "collage1", "rising_noise", 12),
    ("cam-car-park", "collage2", "dimming", 13),
    ("cam-lab", "collage3", "focus_loss", 14),
    ("cam-roof", "collage4", "dirt_buildup", 15),
    ("cam-reception", "collage5", "stable", 16),
]


def _history(args):
    cam, scene, process, seed = args
    recs = []
    for d, ref, fr in timeline.frames(scene, process, DAYS, seed):
        dg = diagnose(fr, ref)
        recs.append({"camera": cam, "t": float(d), "faults": dg.faults(),
                     "metrics": {k: float(v) for k, v in dg.metrics.items()
                                 if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)}, "note": ""})
    return {"camera": cam, "scene": scene, "process": process, "records": recs}


def build(path: Path = CACHE, procs: int = 2) -> list[dict]:
    with Pool(procs) as p:
        fleet = p.map(_history, FLEET)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fleet))
    return fleet


def load(path: Path = CACHE) -> list[dict]:
    return json.loads(Path(path).read_text())


def _health(pp: Passport, now: float) -> float:
    """0-100: 100 = every watched metric still at its baseline, 0 = some metric has reached its fault limit."""
    worst = 0.0
    for w in WATCHES:
        t, y = pp.series(w.metric, until=now)
        if len(t) < 8:
            continue
        base = float(np.median(y[:5]))
        f = pp.forecast(w, now)
        val = f["value_now"] if f.get("value_now") is not None else float(y[-1])
        span = w.threshold - base
        if abs(span) < 1e-9:
            continue
        worst = max(worst, float(np.clip((val - base) / span, 0, 1)))
    return round(100 * (1 - worst), 0)


def rank(fleet: list[dict], now: float, horizon_days: float = 30.0) -> list[dict]:
    rows = []
    for cam in fleet:
        pp = Passport(cam["camera"])
        pp.records = [r for r in cam["records"] if r["t"] <= now]
        fcs = [pp.forecast(w, now) for w in WATCHES]
        exceeded = [f for f in fcs if f["status"] == "exceeded"]
        deg = sorted([f for f in fcs if f["status"] == "degrading" and f["eta_days"] is not None
                      and f["eta_days"] <= horizon_days], key=lambda f: f["eta_days"])
        if exceeded:
            status, eta, fault = "needs attention now", 0.0, exceeded[0]["fault"]
        elif deg:
            status, eta, fault = "maintenance soon", float(deg[0]["eta_days"]), deg[0]["fault"]
        else:
            status, eta, fault = "healthy", None, None
        rows.append({"camera": cam["camera"], "status": status, "fault": fault, "eta_days": eta,
                     "health": _health(pp, now), "simulated_cause": cam["process"]})
    order = {"needs attention now": 0, "maintenance soon": 1, "healthy": 2}
    return sorted(rows, key=lambda r: (order[r["status"]], r["eta_days"] if r["eta_days"] is not None else 1e9, r["health"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--day", type=float, default=30)
    a = ap.parse_args()
    fl = build() if a.build else load()
    for r in rank(fl, a.day):
        print(r)
