"""Evaluation metrics (used by the evaluator, never by the agent)."""
from __future__ import annotations
import cv2
import numpy as np

_QR = None


def _gray_f(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float64)


def ssim(a: np.ndarray, b: np.ndarray, crop: int = 48) -> float:
    """Mean SSIM (Gaussian 11x11, sigma 1.5) on the grayscale central crop."""
    x, y = _gray_f(a), _gray_f(b)
    if crop:
        x, y = x[crop:-crop, crop:-crop], y[crop:-crop, crop:-crop]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m.mean())


def psnr(a: np.ndarray, b: np.ndarray, crop: int = 48) -> float:
    if crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return float(cv2.PSNR(a, b))


def qr_decoded(img: np.ndarray, payloads: list[str]) -> int:
    """How many of the known QR payloads are decoded correctly (downstream task)."""
    global _QR
    if _QR is None:
        _QR = cv2.QRCodeDetectorAruco()
    try:
        ok, texts, _, _ = _QR.detectAndDecodeMulti(img)
    except cv2.error:
        return 0
    if not ok:
        return 0
    return len(set(texts) & set(payloads))


def qr_decoded_roi(img: np.ndarray, payloads: list[str], positions, size: int, margin: int = 12) -> int:
    """Fixed-ROI downstream task: look for each code only where it was at commissioning.

    This is how many real analytics (parking slots, gauges, shelf cameras) work, and it is
    why camera drift silently breaks them.
    """
    global _QR
    if _QR is None:
        _QR = cv2.QRCodeDetectorAruco()
    h, w = img.shape[:2]
    hits = 0
    for (x, y), p in zip(positions, payloads):
        x0, y0 = max(0, x - margin), max(0, y - margin)
        crop = img[y0:min(h, y + size + margin), x0:min(w, x + size + margin)]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            ok, texts, _, _ = _QR.detectAndDecodeMulti(crop)
        except cv2.error:
            continue
        if ok and p in texts:
            hits += 1
    return hits
