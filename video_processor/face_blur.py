from dataclasses import dataclass
from typing import Tuple, Literal, List
import cv2
import numpy as np
from loguru import logger

BlurMethod = Literal["gaussian", "pixelate"]


@dataclass
class FaceAnonymizer:
    cascade_path: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    blur_method: BlurMethod = "gaussian"
    pixelate_blocks: int = 10  # for pixelation
    gaussian_kernel: Tuple[int, int] = (99, 99)
    gaussian_sigma: int = 30
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)

    def __post_init__(self):
        self.detector = cv2.CascadeClassifier(self.cascade_path)
        if self.detector.empty():
            raise FileNotFoundError(f"Failed to load Haar cascade from {self.cascade_path}")
        logger.info(f"Loaded Haar cascade from {self.cascade_path}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(x, y, w, h) for (x, y, w, h) in rects]

    def _blur_roi(self, roi: np.ndarray) -> np.ndarray:
        if self.blur_method == "gaussian":
            return cv2.GaussianBlur(roi, self.gaussian_kernel, self.gaussian_sigma)
        elif self.blur_method == "pixelate":
            h, w = roi.shape[:2]
            x_blocks = max(1, self.pixelate_blocks)
            # downscale then upscale to pixelate
            roi_small = cv2.resize(roi, (x_blocks, max(1, int(x_blocks * h / w))), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unknown blur method: {self.blur_method}")

    def anonymize(self, frame: np.ndarray) -> np.ndarray:
        faces = self.detect_faces(frame)
        for (x, y, w, h) in faces:
            roi = frame[y : y + h, x : x + w]
            frame[y : y + h, x : x + w] = self._blur_roi(roi)
        return frame
