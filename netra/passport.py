"""Camera Health Passport: per-camera history, trends and forecasts.

Every diagnosis is recorded, including values that are still below the fault thresholds. A robust
trend (Theil-Sen slope, Mann-Kendall significance) then answers the maintenance question
"when will this camera cross its fault threshold?" before it actually does.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from .diagnose import Thresholds

_T = Thresholds()


@dataclass(frozen=True)
class Watch:
    metric: str
    threshold: float
    direction: int          # +1: gets worse by rising, -1: gets worse by falling
    fault: str
    cause: str


WATCHES = [
    Watch("sharp_ratio", _T.blur_ratio, -1, "blur", "focus is drifting or the lens is degrading"),
    Watch("mean_ratio", _T.low_light_ratio, -1, "low_light", "the scene is getting darker (lighting or exposure)"),
    Watch("mean_ratio", _T.over_ratio, +1, "overexposure", "the scene is getting brighter (lighting or exposure)"),
    Watch("noise_excess", _T.noise_excess, +1, "noise", "sensor noise is rising (gain, heat or cabling)"),
    Watch("drift_px", _T.drift_px, +1, "drift", "the mount is creeping (loose bracket or vibration)"),
    Watch("dirt_p10", _T.dirt_block_ratio, -1, "lens_dirt", "contamination is building up on the lens"),
]


def mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Return (tau, two-sided p) of the Mann-Kendall trend test (normal approximation, tie-corrected)."""
    n = len(y)
    if n < 4:
        return 0.0, 1.0
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(c * (c - 1) * (2 * c + 5) for c in counts)) / 18.0
    if var <= 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var)
    return float(s / (n * (n - 1) / 2)), float(math.erfc(abs(z) / math.sqrt(2)))


def theil_sen(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust slope and intercept (median of pairwise slopes)."""
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt != 0
    slope = float(np.median((y[j] - y[i])[ok] / dt[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * t))


class Passport:
    def __init__(self, camera_id: str = "cam-1", path: Optional[str] = None):
        self.camera_id = camera_id
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path and self.path.exists():
            self.records = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def record(self, t_days: float, metrics: dict, faults: list[str], note: str = "") -> None:
        rec = {"camera": self.camera_id, "t": float(t_days), "faults": list(faults), "note": note,
               "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))
                           and not isinstance(v, bool)}}
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def series(self, metric: str, until: Optional[float] = None):
        pts = [(r["t"], r["metrics"][metric]) for r in self.records
               if metric in r["metrics"] and (until is None or r["t"] <= until)]
        if not pts:
            return np.array([]), np.array([])
        pts.sort()
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    def forecast(self, w: Watch, now: float, window_days: Optional[float] = None, min_points: int = 8,
                 alpha: float = 0.001, windows: tuple = (10.0, 20.0, 30.0)) -> dict:
        """Theil-Sen trend + Mann-Kendall test. Several look-back windows are tried, shortest first, and the
        first one with a significant worsening trend gives the ETA (accelerating faults such as focus loss
        are flat over a long window but clearly trending over a short one). window_days forces one window."""
        wins = (window_days,) if window_days else windows
        out = {"fault": w.fault, "metric": w.metric, "threshold": w.threshold, "n": 0,
               "status": "insufficient_data", "eta_days": None, "slope_per_day": None}
        t_all, y_all = self.series(w.metric, until=now)
        stable_seen = False
        for win in wins:
            keep = t_all >= now - win
            t, y = t_all[keep], y_all[keep]
            out["n"] = max(out["n"], int(len(t)))
            if len(t) < min_points:
                continue
            slope, icpt = theil_sen(t, y)
            tau, p = mann_kendall(y)
            now_val = icpt + slope * now
            cand = dict(slope_per_day=round(slope, 5), tau=round(tau, 3), p=round(p, 4),
                        value_now=round(now_val, 3), window_days=win)
            if (now_val - w.threshold) * w.direction >= 0:
                out.update(cand, status="exceeded"); return out
            if slope * w.direction > 0 and p < alpha and abs(slope) > 1e-9:
                out.update(cand, status="degrading", eta_days=round((w.threshold - now_val) / slope, 1))
                return out
            if not stable_seen:
                out.update(cand, status="stable"); stable_seen = True
        return out

    def report(self, now: float, horizon_days: float = 30.0, **kw) -> list[dict]:
        """Forecasts that need attention: degrading and expected to cross within the horizon."""
        res = [self.forecast(w, now, **kw) for w in WATCHES]
        out = []
        for w, f in zip(WATCHES, res):
            if f["status"] == "degrading" and f["eta_days"] is not None and f["eta_days"] <= horizon_days:
                f["message"] = (f"{self.camera_id}: {w.cause}. At the current rate it will cross the "
                                f"'{w.fault}' limit in about {f['eta_days']:.0f} days; plan maintenance.")
                out.append(f)
        return sorted(out, key=lambda f: f["eta_days"])
