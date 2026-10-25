"""Slow-degradation simulator: one frame per day for a camera that is slowly going wrong.

Every day the scene itself also changes a little (brightness +-8 %, new objects), exactly like the
evaluation frames, so trends must be extracted from noisy daily measurements.
"""
from __future__ import annotations
import cv2
import numpy as np
from . import scenes

PROCESSES = ["focus_loss", "creeping_drift", "dimming", "rising_noise", "dirt_buildup", "stable"]
FAULT_OF = {"focus_loss": "blur", "creeping_drift": "drift", "dimming": "low_light",
            "rising_noise": "noise", "dirt_buildup": "lens_dirt"}
# per-day rates, chosen on the calibration scenes only
RATES = {"focus_loss": 0.017, "creeping_drift": 0.067, "dimming": 0.0057, "rising_noise": 0.19,
         "dirt_buildup": 0.0027}


def _dirt_mask(shape, seed):
    r = np.random.default_rng(seed)
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    for _ in range(3):
        cv2.ellipse(m, (int(r.integers(100, w - 100)), int(r.integers(100, h - 100))),
                    (int(r.integers(45, 85)), int(r.integers(35, 70))), float(r.uniform(0, 180)), 0, 360, 1.0, -1)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 9), 0, 1)


def degrade(img: np.ndarray, process: str, day: float, rng: np.random.Generator, seed: int = 0) -> np.ndarray:
    d = day
    if process == "stable":
        return img
    if process == "focus_loss":
        s = RATES[process] * d
        return cv2.GaussianBlur(img, (0, 0), s) if s > 0.25 else img
    if process == "creeping_drift":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.007 * d, 1.0)
        M[:, 2] += (RATES[process] * d, 0.5 * RATES[process] * d)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4)
    if process == "dimming":
        return np.clip(img.astype(np.float32) * (1 - RATES[process] * d), 0, 255).astype(np.uint8)
    if process == "rising_noise":
        return np.clip(img.astype(np.float32) + rng.normal(0, RATES[process] * d, img.shape), 0, 255).astype(np.uint8)
    if process == "dirt_buildup":
        a = _dirt_mask(img.shape, seed)[..., None] * min(0.9, RATES[process] * d * 6)
        smear = cv2.GaussianBlur(img, (0, 0), 14).astype(np.float32) + 25
        return np.clip(img.astype(np.float32) * (1 - a) + smear * a, 0, 255).astype(np.uint8)
    raise ValueError(process)


def frames(scene: str, process: str, days: int, seed: int):
    """Yield (day, reference, frame) for day 0..days-1."""
    rng = np.random.default_rng(seed)
    ref = scenes.build_scene(scene)["reference"]
    for d in range(days):
        live = scenes.live_frame(ref, rng)
        yield d, ref, degrade(live, process, d, rng, seed)
