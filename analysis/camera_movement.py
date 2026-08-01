import cv2
import numpy as np


class CameraMovementEstimator:
    """Estimate inter-frame camera shift using sparse optical flow."""

    def __init__(self, max_corners: int = 200):
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self.max_corners = max_corners
        self.cumulative_offset = np.array([0.0, 0.0], dtype=np.float32)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        return pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            return np.array([0.0, 0.0], dtype=np.float32)

        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_pts = self._detect_features(self._prev_gray)

        offset = np.array([0.0, 0.0], dtype=np.float32)
        if self._prev_pts is not None and len(self._prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None
            )
            if next_pts is not None and status is not None:
                good = status.reshape(-1) == 1
                if np.any(good):
                    delta = (next_pts[good] - self._prev_pts[good]).reshape(-1, 2)
                    offset = np.median(delta, axis=0).astype(np.float32)

        self.cumulative_offset += offset
        self._prev_gray = gray
        self._prev_pts = self._detect_features(gray)
        return offset

    def reset(self):
        self._prev_gray = None
        self._prev_pts = None
        self.cumulative_offset = np.array([0.0, 0.0], dtype=np.float32)
