import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamColorAssigner:
    """Assign team_id (0 or 1) via global jersey-color clustering.

    For El Clásico-style kits: saturated jerseys (Barcelona) vs low-saturation white (Real).
    team0 is Barcelona by default (barcelona_team_id=0).
    """

    MODEL_TEAM_MAP = {
        "player_team1": 0,
        "player_team_2": 1,
        "goalkeeper_team1": 0,
        "goalkeeper_team2": 1,
    }

    PLAYER_CLASSES = {
        "player",
        "player_team1",
        "player_team_2",
        "goalkeeper",
        "goalkeeper_team1",
        "goalkeeper_team2",
    }

    def __init__(self, barcelona_team_id: int = 0):
        self.barcelona_team_id = barcelona_team_id
        self.real_team_id = 1 - barcelona_team_id
        self._track_colors: dict[int, list[np.ndarray]] = {}
        self._track_teams: dict[int, int] = {}
        self._fitted = False

    def _extract_jersey_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = max(1, y2 - y1)
        crop = frame[y1 : y1 + h // 2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        pixels = hsv[mask > 0]
        if len(pixels) < 10:
            pixels = hsv.reshape(-1, 3)
        return np.median(pixels, axis=0).astype(np.float32)

    def collect(self, frame: np.ndarray, detections: list[dict]) -> None:
        """Gather jersey colors per track (call on every frame before fit_teams)."""
        for det in detections:
            if det.get("class_name") not in self.PLAYER_CLASSES:
                continue
            track_id = det.get("track_id")
            if track_id is None:
                continue
            if det.get("class_name") in self.MODEL_TEAM_MAP:
                self._track_teams[track_id] = self.MODEL_TEAM_MAP[det["class_name"]]
                continue
            color = self._extract_jersey_color(frame, det["bbox"])
            if color is not None:
                self._track_colors.setdefault(track_id, []).append(color)

    def fit_teams(self, min_samples_per_track: int = 3) -> None:
        """Cluster all tracks globally; map saturated kit -> Barcelona."""
        eligible = {
            tid: np.mean(colors, axis=0)
            for tid, colors in self._track_colors.items()
            if tid not in self._track_teams and len(colors) >= min_samples_per_track
        }
        if len(eligible) < 2:
            self._fitted = True
            return

        track_ids = list(eligible.keys())
        colors = np.array([eligible[tid] for tid in track_ids], dtype=np.float32)
        labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(colors)

        cluster_sizes = [int(np.sum(labels == c)) for c in (0, 1)]
        primary_cluster = 0 if cluster_sizes[0] >= cluster_sizes[1] else 1

        for tid, label in zip(track_ids, labels):
            self._track_teams[tid] = 0 if label == primary_cluster else 1
        self._fitted = True

    def assign_teams(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        if not self._fitted:
            self.collect(frame, detections)
        for det in detections:
            track_id = det.get("track_id")
            if track_id is not None and track_id in self._track_teams:
                det["team_id"] = self._track_teams[track_id]
        return detections
