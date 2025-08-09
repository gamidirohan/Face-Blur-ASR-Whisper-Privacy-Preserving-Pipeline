import numpy as np
import cv2
from video_processor.face_blur import FaceAnonymizer


def _make_roi_with_detail(img_size=100, square_top_left=(30, 30), square_size=40):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    x, y = square_top_left
    s = square_size
    # White square
    cv2.rectangle(img, (x, y), (x + s, y + s), (255, 255, 255), -1)
    # Add a black patch inside the white square to create variance
    pad = 10
    cv2.rectangle(img, (x + s // 2 - pad, y + s // 2 - pad), (x + s // 2 + pad, y + s // 2 + pad), (0, 0, 0), -1)
    roi = img[y : y + s, x : x + s].copy()
    return roi


def test_gaussian_blur_changes_pixels():
    anonymizer = FaceAnonymizer(blur_method="gaussian")
    roi = _make_roi_with_detail()
    blurred = anonymizer._blur_roi(roi)
    assert np.any(blurred != roi)


def test_pixelate_blur_changes_pixels():
    anonymizer = FaceAnonymizer(blur_method="pixelate", pixelate_blocks=6)
    roi = _make_roi_with_detail()
    pixelated = anonymizer._blur_roi(roi)
    assert np.any(pixelated != roi)
