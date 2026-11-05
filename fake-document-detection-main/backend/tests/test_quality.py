from __future__ import annotations

import numpy as np

from modules.ocr.quality import assess


def test_white_scanned_document_background_is_not_mistaken_for_glare():
    import cv2

    image = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    for y in range(120, 1100, 90):
        cv2.line(image, (120, y), (1480, y), (30, 30, 30), 8)

    report = assess(image)

    assert report.clipped_ratio == 0
    assert report.acceptable


def test_black_clipping_still_fails_the_exposure_gate():
    image = np.zeros((1200, 1600, 3), dtype=np.uint8)

    report = assess(image, {"min_sharpness": 0})

    assert not report.acceptable
    assert report.clipped_ratio == 1
    assert "clipped to black" in report.summary
