"""Diagnosis tools: OpenCV-5 measurements that explain *why* a camera feed is degraded.

Every diagnosis is relative to a reference frame captured when the camera was commissioned.
All metrics are chosen to be (roughly) invariant to the other faults:
  * sharpness is normalised by image variance  -> insensitive to exposure changes
  * noise is estimated on the image and divided by its std -> insensitive to gain
  * local-dirt uses block ratios normalised by their median -> insensitive to global blur
  * drift uses ORB + RANSAC on histogram-equalised frames -> insensitive to exposure
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import cv2
import numpy as np

FAULT_PRIORITY = ["drift", "lens_dirt", "low_light", "overexposure", "noise", "blur"]


@dataclass
class Thresholds:
    blur_ratio: float = 0.60        # sharpness index / reference below this -> blur
    low_light_ratio: float = 0.72   # mean luminance / reference below this
    over_ratio: float = 1.30        # mean luminance / reference above this
    over_clip_excess: float = 0.04  # extra fraction of clipped pixels (>=250)
    noise_excess: float = 0.07      # (noise/std) increase over the reference
    drift_px: float = 3.0           # displacement of the image centre
    drift_deg: float = 0.4
    drift_scale: float = 0.008
    drift_min_inliers: int = 25
    drift_min_ratio: float = 0.35
    dirt_block_ratio: float = 0.45  # block sharpness relative to the global median
    dirt_min_blocks: int = 4        # connected blocks needed (4 x 40x40 px)
    dirt_max_fraction: float = 0.45 # more than this = global problem, not local dirt
    block: int = 40


@dataclass
class Finding:
    fault: str
    severity: float                  # 0..1, comparable across fault types
    detail: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    findings: list[Finding]
    metrics: dict
    artifacts: dict = field(default_factory=dict, repr=False)   # drift matrix, dirt mask

    def faults(self) -> list[str]:
        return [f.fault for f in self.findings]

    def severity(self, fault: str) -> float:
        for f in self.findings:
            if f.fault == fault:
                return f.severity
        return 0.0

    def total_severity(self, exclude: tuple = ()) -> float:
        return sum(f.severity for f in self.findings if f.fault not in exclude)

    def to_dict(self) -> dict:
        return {"findings": [asdict(f) for f in self.findings], "metrics": self.metrics}


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness_index(g: np.ndarray) -> float:
    """High-frequency energy / total variance (exposure invariant)."""
    s = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float64)
    lap = cv2.Laplacian(s, cv2.CV_64F)
    return float(lap.var() / (s.var() + 1e-6))


def noise_rel(g: np.ndarray) -> float:
    """Immerkaer fast noise estimate, divided by the image std."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = g.shape
    conv = cv2.filter2D(g.astype(np.float64), -1, k)[1:-1, 1:-1]
    sigma = np.sqrt(np.pi / 2) / 6.0 * np.abs(conv).mean()
    return float(sigma / (g.std() + 1e-6))


def clip_fraction(g: np.ndarray) -> float:
    return float((g >= 250).mean())


def estimate_drift(frame_g: np.ndarray, ref_g: np.ndarray, th: Thresholds) -> dict:
    """ORB + ratio test + RANSAC similarity transform frame -> reference."""
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=8)
    prep = lambda g: cv2.GaussianBlur(cv2.equalizeHist(g), (0, 0), 1.0)
    k1, d1 = orb.detectAndCompute(prep(frame_g), None)
    k2, d2 = orb.detectAndCompute(prep(ref_g), None)
    out = {"confident": False, "inliers": 0, "M": None}
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return out
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < th.drift_min_inliers:
        return out
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None:
        return out
    n_in = int(inl.sum())
    out["inliers"] = n_in
    if n_in < th.drift_min_inliers or n_in / len(good) < th.drift_min_ratio:
        return out
    h, w = frame_g.shape
    centre = np.array([w / 2, h / 2, 1.0])
    disp = float(np.linalg.norm(M @ centre - centre[:2]))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    out.update(confident=True, M=M, disp_px=disp, rot_deg=rot, scale=scale)
    return out


def block_ratios(f_g: np.ndarray, r_g: np.ndarray, bs: int, exclude=None):
    """Per-block Laplacian-variance ratio frame/reference, normalised by its median."""
    lap = lambda g: cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32), cv2.CV_32F) ** 2
    lf, lr = lap(f_g), lap(r_g)
    h, w = f_g.shape
    gh, gw = h // bs, w // bs
    bf = cv2.resize(lf[:gh * bs, :gw * bs], (gw, gh), interpolation=cv2.INTER_AREA)
    br = cv2.resize(lr[:gh * bs, :gw * bs], (gw, gh), interpolation=cv2.INTER_AREA)
    valid = br > np.percentile(br, 25)                 # skip flat reference blocks
    if exclude is not None:                            # sensor saturated / crushed: no contrast to judge
        ex = cv2.resize(exclude[:gh * bs, :gw * bs].astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        valid &= ex < 0.2
    ratio = np.where(valid, bf / (br + 1e-6), np.nan)
    med = np.nanmedian(ratio) if np.isfinite(ratio).any() else 1.0
    return ratio / (med + 1e-9), valid


def detect_dirt(f_g: np.ndarray, r_g: np.ndarray, th: Thresholds, exclude=None):
    ratio, valid = block_ratios(f_g, r_g, th.block, exclude)
    bad = (ratio < th.dirt_block_ratio) & valid
    bad_u8 = bad.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bad_u8, connectivity=8)
    keep = np.zeros_like(bad_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= th.dirt_min_blocks:
            keep[lab == i] = 1
    frac = float(keep.sum() / max(1, valid.sum()))
    vals = ratio[valid & np.isfinite(ratio)]
    p10 = float(np.percentile(vals, 10)) if vals.size else 1.0
    mask = cv2.resize(keep, (f_g.shape[1], f_g.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # per-pixel estimate of how much local contrast the smudge removed: contrast ~ (1 - alpha)
    r_full = cv2.resize(np.nan_to_num(ratio, nan=1.0).astype(np.float32), (f_g.shape[1], f_g.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 10)
    alpha = np.clip(1 - np.sqrt(np.clip(r_full, 0, 1)), 0, 0.92) * np.clip(feather, 0, 1)
    return frac, mask, alpha, p10


def saturation_mask(frame: np.ndarray) -> np.ndarray:
    """Pixels where the sensor clipped (>=250) or crushed (<=4) in any channel, slightly dilated."""
    m = ((frame >= 250) | (frame <= 4)).any(axis=2) if frame.ndim == 3 else ((frame >= 250) | (frame <= 4))
    return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))


def diagnose(frame: np.ndarray, ref: np.ndarray, th: Thresholds | None = None,
             exclude_mask: np.ndarray | None = None) -> Diagnosis:
    th = th or Thresholds()
    fg, rg = _gray(frame), _gray(ref)
    m: dict = {}
    findings: list[Finding] = []
    art: dict = {}

    # --- drift ---------------------------------------------------------------------
    d = estimate_drift(fg, rg, th)
    m["drift_confident"], m["drift_inliers"] = d["confident"], d["inliers"]
    if d["confident"]:
        m.update(drift_px=round(d["disp_px"], 2), drift_deg=round(d["rot_deg"], 3),
                 drift_scale=round(d["scale"], 4))
        if (d["disp_px"] > th.drift_px or abs(d["rot_deg"]) > th.drift_deg
                or abs(d["scale"] - 1) > th.drift_scale):
            sev = min(1.0, d["disp_px"] / 40 + abs(d["rot_deg"]) / 6 + abs(d["scale"] - 1) * 8)
            findings.append(Finding("drift", sev, {k: m[k] for k in ("drift_px", "drift_deg", "drift_scale")}))
            art["drift_M"] = d["M"]

    # --- exposure ------------------------------------------------------------------
    mean_ratio = float(fg.mean() / (rg.mean() + 1e-6))
    clip_excess = clip_fraction(fg) - clip_fraction(rg)
    m.update(mean_ratio=round(mean_ratio, 3), clip_excess=round(clip_excess, 4))
    if mean_ratio < th.low_light_ratio:
        findings.append(Finding("low_light", 1 - mean_ratio, {"mean_ratio": m["mean_ratio"]}))
    elif mean_ratio > th.over_ratio or clip_excess > th.over_clip_excess:
        findings.append(Finding("overexposure", min(1.0, max(mean_ratio - 1, 0) + max(clip_excess, 0)),
                                {"mean_ratio": m["mean_ratio"], "clip_excess": m["clip_excess"]}))

    # --- noise -----------------------------------------------------------------------
    n_ex = noise_rel(fg) - noise_rel(rg)
    m["noise_excess"] = round(n_ex, 4)
    if n_ex > th.noise_excess:
        findings.append(Finding("noise", min(1.0, n_ex * 2.5), {"noise_excess": m["noise_excess"]}))

    # --- blur (global) -----------------------------------------------------------------
    sr = sharpness_index(fg) / (sharpness_index(rg) + 1e-9)
    m["sharp_ratio"] = round(sr, 3)
    if sr < th.blur_ratio:
        findings.append(Finding("blur", 1 - sr, {"sharp_ratio": m["sharp_ratio"]}))

    # --- lens dirt (local) -------------------------------------------------------------
    # Block ratios are only trustworthy when no global fault distorts them (blur, clipping
    # or noise make sharp regions such as QR codes collapse faster than the median block),
    # so dirt is judged last, after the global faults are fixed. The agent re-diagnoses
    # after every step, so it still gets found.
    if not findings:
        frac, mask, alpha, p10 = detect_dirt(fg, rg, th, exclude_mask)
        m["dirt_p10"] = round(p10, 3)
        m["dirt_fraction"] = round(frac, 4)
        if frac > 0 and frac <= th.dirt_max_fraction:
            findings.append(Finding("lens_dirt", min(1.0, frac * 4), {"dirt_fraction": m["dirt_fraction"],
                                                        "mask_px": int(mask.sum())}))
            art["dirt_mask"], art["dirt_alpha"] = mask, alpha
    findings.sort(key=lambda f: FAULT_PRIORITY.index(f.fault))
    return Diagnosis(findings, m, art)
