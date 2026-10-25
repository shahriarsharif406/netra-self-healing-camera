"""Fault simulator: realistic camera degradations with a severity in [0, 1]."""
from __future__ import annotations
import cv2
import numpy as np

FAULTS = ["blur", "low_light", "overexposure", "noise", "drift", "lens_dirt"]


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img, rng, sev):
    sigma = 1.0 + 3.5 * sev
    return cv2.GaussianBlur(img, (0, 0), sigma), {"sigma": round(sigma, 2)}


def low_light(img, rng, sev):
    gamma, gain = 1.4 + 0.6 * sev, 0.60 - 0.30 * sev
    out = 255.0 * gain * (img.astype(np.float32) / 255.0) ** gamma
    return _u8(np.round(out)), {"gamma": round(gamma, 2), "gain": round(gain, 2)}


def overexposure(img, rng, sev):
    gain, off = 1.45 + 0.75 * sev, 20 + 25 * sev
    return _u8(img.astype(np.float32) * gain + off), {"gain": round(gain, 2), "offset": round(off, 1)}


def noise(img, rng, sev):
    sigma = 16 + 22 * sev
    n = rng.normal(0, sigma, img.shape).astype(np.float32)
    return _u8(img.astype(np.float32) + n), {"sigma": round(sigma, 1)}


def drift(img, rng, sev):
    h, w = img.shape[:2]
    tx, ty = (rng.choice([-1, 1]) * (10 + 22 * sev) * rng.uniform(0.7, 1.0) for _ in range(2))
    rot = rng.choice([-1, 1]) * (1.0 + 2.5 * sev)
    sc = 1.0 + rng.choice([-1, 1]) * (0.01 + 0.025 * sev)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, sc)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return out, {"tx": round(float(tx), 1), "ty": round(float(ty), 1), "rot": round(float(rot), 2),
                 "scale": round(float(sc), 3)}


def lens_dirt(img, rng, sev):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    n = int(rng.integers(2, 4))
    for _ in range(n):
        c = (int(rng.integers(100, w - 100)), int(rng.integers(100, h - 100)))
        ax = (int(rng.integers(45, 85)), int(rng.integers(35, 70)))
        cv2.ellipse(mask, c, ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 9)
    alpha = np.clip(mask, 0, 1)[..., None] * (0.65 + 0.3 * sev)
    smeared = cv2.GaussianBlur(img, (0, 0), 14).astype(np.float32) + 25
    out = img.astype(np.float32) * (1 - alpha) + smeared * alpha
    return _u8(out), {"blobs": n, "strength": round(0.65 + 0.3 * sev, 2)}


_FN = {"blur": blur, "low_light": low_light, "overexposure": overexposure, "noise": noise,
       "drift": drift, "lens_dirt": lens_dirt}
# physical order of a real pipeline: pose -> lens -> optics -> exposure -> sensor noise
_ORDER = ["drift", "lens_dirt", "blur", "low_light", "overexposure", "noise"]


def apply_faults(img: np.ndarray, faults: list[str], rng: np.random.Generator,
                 sev: float | None = None):
    """Apply one or more faults in physical order. Returns (image, parameters)."""
    params = {}
    for f in sorted(faults, key=_ORDER.index):
        s = float(rng.uniform(0.2, 1.0)) if sev is None else sev
        img, p = _FN[f](img, rng, s)
        params[f] = {"severity": round(s, 2), **p}
    return img, params
