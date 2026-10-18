"""Image preprocessing for scanned / photographed administrative documents.

Pipeline: decode -> grayscale -> upscale small images -> denoise (NL-means)
-> binarisation (Sauvola) + speck removal -> deskew (projection profile). Each step can be toggled so its effect on OCR
can be measured in the evaluation (ablation study).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    denoise: bool = True
    clahe: bool = False  # see preprocess(): harmful before binarisation
    deskew: bool = True
    binarize: bool = True
    target_min_width: int = 1600  # Tesseract works best around 300 dpi


@dataclass
class PreprocessResult:
    image: np.ndarray
    skew_angle: float = 0.0
    steps: list[str] = field(default_factory=list)


def load_pages(data: bytes, filename: str = "") -> list[np.ndarray]:
    """Decode an uploaded file (image or PDF) into a list of BGR pages."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(io.BytesIO(data))
        pages = []
        for page in pdf:
            pil = page.render(scale=300 / 72).to_pil().convert("RGB")
            pages.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        return pages

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupted image file")
    return [img]


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale(gray: np.ndarray, min_width: int) -> np.ndarray:
    h, w = gray.shape
    if w >= min_width:
        return gray
    scale = min_width / w
    return cv2.resize(gray, (min_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)


def denoise(gray: np.ndarray) -> np.ndarray:
    # Non-local means keeps character edges sharper than a Gaussian or median filter
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0, step: float = 0.25) -> float:
    """Estimate the text skew angle (degrees) with a projection-profile search.

    For each candidate angle the binarised image is rotated and the variance of the
    horizontal projection is computed: text lines aligned with the rows give sharp
    peaks, hence maximal variance.

    The ink mask comes from Sauvola (local) thresholding: a global Otsu threshold
    turns the dark side of an unevenly lit photo into "ink" and biases the angle.
    """
    binary = gray if is_binary(gray) else remove_specks(sauvola(gray))
    bw = 255 - binary
    # Work on a reduced image for speed
    scale = 800 / max(bw.shape)
    if scale < 1:
        bw = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = bw.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rotated.sum(axis=1, dtype=np.float64)))

    # Coarse search, then refine around the best coarse angle
    coarse = np.arange(-max_angle, max_angle + 1e-9, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, step)
    return float(max(fine, key=score))


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(gray, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=255)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Sauvola local thresholding: T = m * (1 + k * (s / R - 1)).

    More robust than a global Otsu threshold to uneven lighting (phone photos, shadows).
    Local mean and std are computed with box filters (integral images).
    """
    img = gray.astype(np.float64)
    mean = cv2.boxFilter(img, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img * img, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    threshold = mean * (1 + k * (std / r - 1))
    return np.where(img > threshold, 255, 0).astype(np.uint8)


def is_binary(img: np.ndarray) -> bool:
    return img.dtype == np.uint8 and np.isin(img[::7, ::7], (0, 255)).all()


def remove_specks(binary: np.ndarray, min_area: int = 12) -> np.ndarray:
    """Remove isolated black components smaller than `min_area` pixels (salt noise)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < min_area
    small[0] = False  # label 0 is the background
    out = binary.copy()
    out[small[labels]] = 255
    return out


def preprocess(img: np.ndarray, opts: PreprocessOptions | None = None) -> PreprocessResult:
    opts = opts or PreprocessOptions()
    steps = ["grayscale"]
    gray = to_gray(img)
    gray = upscale(gray, opts.target_min_width)
    if opts.denoise:
        gray = denoise(gray)
        steps.append("denoise")
    if opts.clahe:
        # Only useful without binarisation: CLAHE amplifies background noise,
        # which Sauvola then turns into speckle.
        gray = enhance_contrast(gray)
        steps.append("clahe")
    if opts.binarize:
        gray = remove_specks(sauvola(gray))
        steps.append("sauvola")
    angle = 0.0
    if opts.deskew:
        angle = estimate_skew(gray)
        if abs(angle) > 0.1:
            binary = is_binary(gray)
            gray = rotate(gray, angle)
            if binary:  # interpolation creates grey edge pixels
                gray = np.where(gray > 127, 255, 0).astype(np.uint8)
        steps.append(f"deskew({angle:+.2f}°)")
    return PreprocessResult(image=gray, skew_angle=angle, steps=steps)
