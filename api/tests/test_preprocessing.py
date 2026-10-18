import cv2
import numpy as np
import pytest

from api.app.preprocessing import estimate_skew, preprocess, rotate, sauvola


def text_like_page(h=1200, w=900) -> np.ndarray:
    img = np.full((h, w), 255, np.uint8)
    for y in range(100, h - 100, 40):
        cv2.putText(img, "LOREM IPSUM DOLOR SIT AMET 2026", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return img


@pytest.mark.parametrize("angle", [-6.0, -2.5, 0.0, 3.0, 7.5])
def test_estimate_skew_recovers_rotation(angle):
    page = text_like_page()
    skewed = rotate(page, angle)
    # Rotating by -estimate should undo the skew, so estimate ≈ -angle
    assert estimate_skew(skewed) == pytest.approx(-angle, abs=0.5)


def test_sauvola_handles_uneven_lighting():
    page = text_like_page().astype(np.float64)
    gradient = np.linspace(1.0, 0.45, page.shape[1])[None, :]
    dark = (page * gradient).astype(np.uint8)
    bw = sauvola(dark)
    right_half = bw[:, bw.shape[1] // 2:]
    # Background on the dark side must stay white (Otsu would turn much of it black)
    assert (right_half == 255).mean() > 0.85
    assert set(np.unique(bw)) <= {0, 255}


def test_preprocess_outputs_binary_image():
    result = preprocess(cv2.cvtColor(rotate(text_like_page(), 4), cv2.COLOR_GRAY2BGR))
    assert result.image.ndim == 2
    assert result.skew_angle == pytest.approx(-4, abs=0.5)
    assert result.steps[0] == "grayscale" and "sauvola" in result.steps
    assert set(np.unique(result.image)) <= {0, 255}
