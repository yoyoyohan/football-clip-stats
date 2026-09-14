import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamColorAssigner:
    """Assign team_id (0 or 1) via global jersey-color clustering.

    Convention after ``fit_teams`` (unless ``barcelona_team_id`` swaps labels):
    **team0** = brighter / more chromatic kit; **team1** = darker / less
    chromatic kit (e.g. yellow → 0, black → 1). Per-track assignments stay
    sticky after fit.
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

    # Never use these for jersey clustering / team assignment.
    EXCLUDED_CLASSES = {
        "referee",
        "ball",
        "goalpost",
        "score",
        "gametime",
    }

    def __init__(self, barcelona_team_id: int = 0):
        # API compat: when 1, swap the brightness-based cluster labels.
        self.barcelona_team_id = barcelona_team_id
        self.real_team_id = 1 - barcelona_team_id
        self._track_colors: dict[int, list[np.ndarray]] = {}
        self._track_teams: dict[int, int] = {}
        self._fitted = False

    def _torso_crop(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        # Upper half (jersey), central 70% width to reduce grass / sleeves.
        top = y1
        bottom = y1 + max(1, h // 2)
        inset = int(0.15 * w)
        left = x1 + inset
        right = max(left + 1, x2 - inset)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return crop

    def _extract_jersey_features(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """Features that separate dark and bright kits (e.g. black vs yellow).

        Does **not** require S≥30 for dark pixels — that discarded black jerseys.
        Feature vector: [mean_V, mean_S, mean_chroma, mean_L] in float32.
        """
        crop = self._torso_crop(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Keep chromatic pixels OR dark (low-V) pixels — black kits are low-S.
        dark = v <= 90
        chromatic = s >= 25
        mask = dark | chromatic
        if int(np.count_nonzero(mask)) < 10:
            mask = np.ones(v.shape, dtype=bool)

        hsv_px = hsv[mask]
        bgr_px = crop[mask]
        lab_px = lab[mask]
        if len(hsv_px) < 5:
            return None

        mean_v = float(np.mean(hsv_px[:, 2]))
        mean_s = float(np.mean(hsv_px[:, 1]))
        chroma = float(np.mean(np.max(bgr_px, axis=1) - np.min(bgr_px, axis=1)))
        mean_l = float(np.mean(lab_px[:, 0]))
        return np.array([mean_v, mean_s, chroma, mean_l], dtype=np.float32)

    # Back-compat alias used by older call sites / tests.
    def _extract_jersey_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        return self._extract_jersey_features(frame, bbox)

    def collect(self, frame: np.ndarray, detections: list[dict]) -> None:
        """Gather jersey features per track (call on every frame before fit_teams)."""
        for det in detections:
            class_name = det.get("class_name")
            if class_name in self.EXCLUDED_CLASSES:
                continue
            if class_name not in self.PLAYER_CLASSES:
                continue
            track_id = det.get("track_id")
            if track_id is None:
                continue
            if class_name in self.MODEL_TEAM_MAP:
                self._track_teams[track_id] = self.MODEL_TEAM_MAP[class_name]
                continue
            feats = self._extract_jersey_features(frame, det["bbox"])
            if feats is not None:
                self._track_colors.setdefault(track_id, []).append(feats)

    def fit_teams(self, min_samples_per_track: int = 3) -> None:
        """Cluster eligible tracks; label by brightness/chroma (stable vs random size)."""
        eligible = {
            tid: np.mean(colors, axis=0)
            for tid, colors in self._track_colors.items()
            if tid not in self._track_teams and len(colors) >= min_samples_per_track
        }
        if len(eligible) < 2:
            # Too few tracks (dummy / empty clips) — leave team_ids unset.
            self._fitted = True
            return

        track_ids = list(eligible.keys())
        features = np.array([eligible[tid] for tid in track_ids], dtype=np.float32)
        labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(features)

        # Stable labeling: brighter / more chromatic → team0; darker → team1.
        # features columns: V, S, chroma, L
        scores = []
        for c in (0, 1):
            subset = features[labels == c]
            if len(subset) == 0:
                scores.append(-1e9)
                continue
            mean_v = float(np.mean(subset[:, 0]))
            mean_s = float(np.mean(subset[:, 1]))
            mean_chroma = float(np.mean(subset[:, 2]))
            mean_l = float(np.mean(subset[:, 3]))
            scores.append(mean_v + mean_l + 0.5 * mean_s + 0.75 * mean_chroma)

        bright_cluster = 0 if scores[0] >= scores[1] else 1
        # Map bright_cluster → logical team0, then optional barcelona swap.
        for tid, label in zip(track_ids, labels):
            logical = 0 if label == bright_cluster else 1
            if self.barcelona_team_id == 1:
                logical = 1 - logical
            self._track_teams[tid] = logical
        self._fitted = True

    def assign_teams(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        if not self._fitted:
            self.collect(frame, detections)
        for det in detections:
            if det.get("class_name") in self.EXCLUDED_CLASSES:
                continue
            track_id = det.get("track_id")
            if track_id is not None and track_id in self._track_teams:
                det["team_id"] = self._track_teams[track_id]
        return detections
