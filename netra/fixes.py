"""Fix tools. Each tool takes (img, ctx) and returns (new_img, params).

ctx carries what the agent knows: the reference frame and its statistics, the current
diagnosis artefacts (drift matrix, dirt mask) and the noise level. Tools never modify the
input array in place, so the agent can always roll back.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import cv2
import numpy as np
from .diagnose import sharpness_index, noise_rel, _gray


@dataclass(frozen=True)
class Tool:
    name: str
    fault: str
    risk: str            # "low" = reversible, pixel-level, no geometry change; "medium" = needs approval
    fn: Callable
    description: str


# ---------------------------------------------------------------- exposure ------------
def _solve_gamma(g: np.ndarray, target_mean: float) -> float:
    x = np.clip(g.astype(np.float32) / 255.0, 1e-4, 1.0)
    lo, hi = 0.15, 6.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if 255.0 * float((x ** mid).mean()) > target_mean:
            lo = mid               # still too bright -> larger gamma darkens
        else:
            hi = mid
    return (lo + hi) / 2


def _gamma_lut(gamma: float) -> np.ndarray:
    return np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)


def exposure_gamma(img, ctx):
    g = _gray(img)
    gamma = _solve_gamma(g, ctx["ref_mean"])
    return cv2.LUT(img, _gamma_lut(gamma)), {"gamma": round(gamma, 3)}


def exposure_clahe(img, ctx):
    gamma = _solve_gamma(_gray(img), ctx["ref_mean"])
    lab = cv2.cvtColor(cv2.LUT(img, _gamma_lut(gamma ** 0.7)), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), {"gamma": round(gamma ** 0.7, 3), "clip": 2.0}


def exposure_linear(img, ctx):
    gain = ctx["ref_mean"] / (float(_gray(img).mean()) + 1e-6)
    return cv2.convertScaleAbs(img, alpha=gain, beta=0), {"gain": round(gain, 3)}


# ------------------------------------------------------------------- noise ------------
def _sigma_ch(g: np.ndarray) -> float:
    """Per-colour-channel noise sigma, estimated on the grey image (grey averages the channel
    noise, std factor ~0.67 for BGR->grey weights)."""
    return noise_rel(g) * float(g.std()) / 0.67


def denoise_nlm(img, ctx):
    sigma = _sigma_ch(_gray(img))
    h = float(np.clip(1.2 * sigma, 3, 45))
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21), {"h": round(h, 1), "sigma_est": round(sigma, 1)}


def denoise_bilateral(img, ctx):
    sigma = _sigma_ch(_gray(img))
    sc = float(np.clip(2.0 * sigma, 15, 90))
    out = img
    for _ in range(3):
        out = cv2.bilateralFilter(out, 7, sc, 5)
    return out, {"sigma_color": round(sc, 1), "passes": 3, "sigma_est": round(sigma, 1)}


# -------------------------------------------------------------------- blur ------------
K_WIENER = 0.02   # tuned on the calibration scenes only


def _wiener(ch: np.ndarray, sigma: float, k: float) -> np.ndarray:
    """Wiener deconvolution of a Gaussian PSF with cv2.dft (edge-replicated padding)."""
    h, w = ch.shape
    pad = int(4 * sigma) + 8
    p = cv2.copyMakeBorder(ch.astype(np.float32) / 255.0, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    H, W = cv2.getOptimalDFTSize(p.shape[0]), cv2.getOptimalDFTSize(p.shape[1])
    p = cv2.copyMakeBorder(p, 0, H - p.shape[0], 0, W - p.shape[1], cv2.BORDER_REPLICATE)
    fy = np.fft.fftfreq(H).astype(np.float32)[:, None]
    fx = np.fft.fftfreq(W).astype(np.float32)[None, :]
    otf = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    filt = (otf / (otf ** 2 + k)).astype(np.float32)
    spec = cv2.dft(p, flags=cv2.DFT_COMPLEX_OUTPUT)
    spec *= filt[..., None]
    out = cv2.idft(spec, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    return out[pad:pad + h, pad:pad + w] * 255.0


def deblur_wiener(img, ctx):
    """Search the PSF width that restores the reference sharpness (no ground truth needed)."""
    g = _gray(img)
    best, best_err = None, 1e9
    for sigma in (1.0, 1.5, 2.0, 2.6, 3.2, 4.0, 4.8):
        cand = np.clip(_wiener(g, sigma, K_WIENER), 0, 255).astype(np.uint8)
        ratio = sharpness_index(cand) / (ctx["ref_sharp"] + 1e-9)
        ring = float(((_wiener(g, sigma, K_WIENER) < -8) | (_wiener(g, sigma, K_WIENER) > 263)).mean())
        err = abs(np.log(max(ratio, 1e-3))) + 20 * ring
        if err < best_err:
            best, best_err = sigma, err
    chans = [np.clip(_wiener(img[:, :, c], best, K_WIENER), 0, 255) for c in range(3)]
    return np.dstack(chans).astype(np.uint8), {"psf_sigma": best}


def deblur_unsharp(img, ctx):
    g = _gray(img)
    best, best_err = (1.0, 0.5), 1e9
    for sigma in (1.0, 2.0, 3.0):
        for amount in (0.6, 1.2, 2.0):
            b = cv2.GaussianBlur(g, (0, 0), sigma)
            cand = cv2.addWeighted(g, 1 + amount, b, -amount, 0)
            err = abs(np.log(max(sharpness_index(cand) / (ctx["ref_sharp"] + 1e-9), 1e-3)))
            if err < best_err:
                best, best_err = (sigma, amount), err
    sigma, amount = best
    return cv2.addWeighted(img, 1 + amount, cv2.GaussianBlur(img, (0, 0), sigma), -amount, 0), \
        {"sigma": sigma, "amount": amount}


# ------------------------------------------------------------------- drift ------------
def align_orb(img, ctx):
    M = ctx.get("drift_M")
    if M is None:
        raise RuntimeError("no drift transform from diagnosis")
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return out, {"dx": round(float(M[0, 2]), 1), "dy": round(float(M[1, 2]), 1),
                 "rot_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2)}


def align_ecc(img, ctx):
    """Coarse-to-fine ECC alignment (independent of feature matching)."""
    prep = lambda g: cv2.GaussianBlur(cv2.equalizeHist(g), (0, 0), 1.5)
    ref_g, f_g = prep(_gray(ctx["ref"])), prep(_gray(img))
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for s in (0.25, 0.5, 1.0):
        rs = cv2.resize(ref_g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
        fs = cv2.resize(f_g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
        w_s = warp.copy()
        w_s[:, 2] *= s
        _, w_s = cv2.findTransformECC(rs, fs, w_s, cv2.MOTION_AFFINE, crit, None, 5)
        warp = w_s
        warp[:, 2] /= s
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT)
    return out, {"dx": round(float(warp[0, 2]), 1), "dy": round(float(warp[1, 2]), 1)}


# -------------------------------------------------------------------- dirt ------------
def _inpaint(img, ctx, flag, name):
    mask = ctx.get("dirt_mask")
    if mask is None or mask.sum() == 0:
        raise RuntimeError("no dirt mask from diagnosis")
    return cv2.inpaint(img, mask, 7, flag), {"method": name, "masked_px": int(mask.sum())}


def dehaze_local(img, ctx):
    """Undo a translucent smudge: observed ~ (1-a)*scene + a*(blur+haze).

    High frequencies are scaled back by 1/(1-a) and the haze offset removed inside the mask.
    Needs no content to be invented, unlike inpainting. The model assumes a smear that mostly
    removes local contrast and adds a brightness offset.
    """
    alpha = ctx.get("dirt_alpha")
    if alpha is None or alpha.max() <= 0:
        raise RuntimeError("no dirt alpha map from diagnosis")
    f = img.astype(np.float32)
    lp = cv2.GaussianBlur(f, (0, 0), 10)
    a = alpha[..., None]
    out = lp - HAZE_OFFSET * a + (f - lp) / (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8), {"mean_alpha_in_mask": round(float(alpha[alpha > 0].mean()), 3)}


HAZE_OFFSET = 25.0


def inpaint_telea(img, ctx):
    return _inpaint(img, ctx, cv2.INPAINT_TELEA, "telea")


def inpaint_ns(img, ctx):
    return _inpaint(img, ctx, cv2.INPAINT_NS, "navier-stokes")


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("align_orb", "drift", "medium", align_orb, "Warp back with the ORB/RANSAC similarity transform"),
    Tool("align_ecc", "drift", "medium", align_ecc, "Coarse-to-fine ECC affine alignment"),
    Tool("dehaze_local", "lens_dirt", "medium", dehaze_local, "Restore local contrast inside smudged regions"),
    Tool("inpaint_telea", "lens_dirt", "medium", inpaint_telea, "Inpaint smudged regions (Telea)"),
    Tool("inpaint_ns", "lens_dirt", "medium", inpaint_ns, "Inpaint smudged regions (Navier-Stokes)"),
    Tool("exposure_gamma", "exposure", "low", exposure_gamma, "Gamma curve that matches reference brightness"),
    Tool("exposure_clahe", "exposure", "low", exposure_clahe, "Gamma + CLAHE on the L channel"),
    Tool("exposure_linear", "exposure", "low", exposure_linear, "Linear gain to reference brightness"),
    Tool("denoise_nlm", "noise", "low", denoise_nlm, "Non-local-means denoising"),
    Tool("denoise_bilateral", "noise", "low", denoise_bilateral, "Edge-preserving bilateral filter"),
    Tool("deblur_wiener", "blur", "low", deblur_wiener, "Wiener deconvolution, PSF width searched against the reference"),
    Tool("deblur_unsharp", "blur", "low", deblur_unsharp, "Unsharp masking tuned against the reference"),
]}

CANDIDATES = {
    "drift": ["align_orb", "align_ecc"],
    "lens_dirt": ["dehaze_local", "inpaint_telea"],
    "low_light": ["exposure_clahe", "exposure_gamma"],
    "overexposure": ["exposure_linear", "exposure_gamma"],
    "noise": ["denoise_bilateral", "denoise_nlm"],
    "blur": ["deblur_wiener", "deblur_unsharp"],
}
