"""Offline test scenes: bundled scikit-image photos + three QR codes as a downstream task.

A "pair" is (reference, live_clean):
  * reference  - the commissioning frame the agent stores for a fixed camera
  * live_clean - the same scene a bit later (small brightness change + objects that
                 were not in the reference). Faults are applied to live_clean, so the
                 agent can never see the ground truth, and its reference is NOT identical
                 to the ground truth.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage import data

W, H = 640, 480
CALIB_SCENES = ["astronaut", "coins", "gravel", "hubble_deep_field"]
TEST_SCENES = ["chelsea", "coffee", "rocket", "camera", "moon", "brick", "grass",
               "immunohistochemistry"]
# Fresh scenes, created AFTER the first evaluation round: random 6-patch collages that also draw
# on four photos that were never used for tuning or for the first test round.
FRESH_SCENES = [f"collage{i}" for i in range(8)]
_COLLAGE_POOL = CALIB_SCENES + TEST_SCENES + ["rocket", "retina", "cell", "page", "text"]
QR_SIZE = 104          # px, including the white quiet zone
QR_MARGIN = 64         # keep codes away from the border so drift does not cut them
QR_POSITIONS = [(QR_MARGIN, QR_MARGIN),
                (W - QR_MARGIN - QR_SIZE, QR_MARGIN),
                (W - QR_MARGIN - QR_SIZE, H - QR_MARGIN - QR_SIZE)]


def _load(name: str) -> np.ndarray:
    img = getattr(data, name)()
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _cover_resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(W / w, H / h)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    y0 = (img.shape[0] - H) // 2
    x0 = (img.shape[1] - W) // 2
    return img[y0:y0 + H, x0:x0 + W].copy()


def make_qr(payload: str) -> np.ndarray:
    q = cv2.QRCodeEncoder.create().encode(payload)
    inner = QR_SIZE - 16
    q = cv2.resize(q, (inner, inner), interpolation=cv2.INTER_NEAREST)
    q = cv2.copyMakeBorder(q, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)


def _collage(seed: int) -> np.ndarray:
    rng = np.random.default_rng(7000 + seed)
    canvas = np.zeros((H, W, 3), np.uint8)
    cw, ch = W // 3, H // 2
    for j in range(6):
        src = _cover_resize(_load(str(rng.choice(_COLLAGE_POOL))))
        sz = int(rng.integers(260, 420))
        x0, y0 = int(rng.integers(0, W - sz)), int(rng.integers(0, H - min(sz, H - 1)))
        patch = cv2.resize(src[y0:y0 + min(sz, H - y0), x0:x0 + sz], (cw, ch), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        canvas[(j // 3) * ch:(j // 3 + 1) * ch, (j % 3) * cw:(j % 3 + 1) * cw] = patch
    return canvas


def build_scene(name: str) -> dict:
    base = _collage(int(name[len("collage"):])) if name.startswith("collage") else _cover_resize(_load(name))
    payloads = [f"NETRA-{name}-{c}" for c in "ABC"]
    for (x, y), p in zip(QR_POSITIONS, payloads):
        base[y:y + QR_SIZE, x:x + QR_SIZE] = make_qr(p)
    return {"name": name, "reference": base, "payloads": payloads}


def _qr_boxes():
    return [(x, y, x + QR_SIZE, y + QR_SIZE) for x, y in QR_POSITIONS]


def live_frame(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same scene later: +-8% brightness and 2-3 new 'objects' (never over a QR code)."""
    img = reference.astype(np.float32) * rng.uniform(0.92, 1.08)
    boxes = _qr_boxes()
    placed = 0
    while placed < int(rng.integers(2, 4)):
        w, h = int(rng.integers(30, 70)), int(rng.integers(30, 70))
        x, y = int(rng.integers(100, W - 100 - w)), int(rng.integers(100, H - 100 - h))
        if any(x < bx1 + 20 and x + w > bx0 - 20 and y < by1 + 20 and y + h > by0 - 20
               for bx0, by0, bx1, by1 in boxes):
            continue
        img[y:y + h, x:x + w] = rng.integers(20, 235, size=3)
        placed += 1
    return np.clip(img, 0, 255).astype(np.uint8)


def make_pair(name: str, rng: np.random.Generator):
    s = build_scene(name)
    return s["reference"], live_frame(s["reference"], rng), s["payloads"]
