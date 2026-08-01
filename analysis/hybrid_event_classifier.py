"""Hybrid event classifier: geometry proposes, ML confirms (optional)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.event_features import (
    GoalCandidateFeatures,
    PassCandidateFeatures,
    ShotCandidateFeatures,
)


DEFAULT_MODEL_DIR = Path("models/event_classifiers")


@dataclass
class ClassificationResult:
    accepted: bool
    probability: float
    source: str  # geometry | model | geometry+model


class HybridEventClassifier:
    """Score event candidates using a trained sklearn model when available.

  Without trained weights this acts as a pass-through (geometry only).
  Train with ``training/train_event_classifier.py`` once you have labeled clips.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, threshold: float = 0.5):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self._models: dict[str, object] = {}
        self._load_models()

    def _load_models(self) -> None:
        try:
            import joblib
        except ImportError:
            return

        for event_type in ("pass", "goal", "shot"):
            path = self.model_dir / f"{event_type}_classifier.joblib"
            if path.exists():
                self._models[event_type] = joblib.load(path)

    @property
    def has_pass_model(self) -> bool:
        return "pass" in self._models

    def _predict(self, event_type: str, features: list[float]) -> tuple[float, str]:
        model = self._models.get(event_type)
        if model is None:
            return 1.0, "geometry"

        proba = float(model.predict_proba(np.array([features]))[0][1])
        return proba, "model"

    def classify_pass(self, features: PassCandidateFeatures) -> ClassificationResult:
        vector = features.to_vector()
        geometry_ok = features.geometry_score >= 0.5

        if "pass" not in self._models:
            return ClassificationResult(accepted=geometry_ok, probability=features.geometry_score, source="geometry")

        prob, source = self._predict("pass", vector)
        accepted = geometry_ok and prob >= self.threshold
        return ClassificationResult(accepted=accepted, probability=prob, source=f"geometry+{source}")

    def classify_goal(self, features: GoalCandidateFeatures) -> ClassificationResult:
        vector = features.to_vector()
        geometry_ok = features.geometry_score >= 0.5

        if "goal" not in self._models:
            return ClassificationResult(accepted=geometry_ok, probability=features.geometry_score, source="geometry")

        prob, source = self._predict("goal", vector)
        accepted = geometry_ok and prob >= self.threshold
        return ClassificationResult(accepted=accepted, probability=prob, source=f"geometry+{source}")

    def classify_shot(self, features: ShotCandidateFeatures) -> ClassificationResult:
        vector = features.to_vector()
        geometry_ok = features.geometry_score >= 0.5

        if "shot" not in self._models:
            return ClassificationResult(accepted=geometry_ok, probability=features.geometry_score, source="geometry")

        prob, source = self._predict("shot", vector)
        accepted = geometry_ok and prob >= self.threshold
        return ClassificationResult(accepted=accepted, probability=prob, source=f"geometry+{source}")

    def export_training_schema(self, path: str | Path) -> None:
        schema = {
            "pass": PassCandidateFeatures.feature_names(),
            "goal": GoalCandidateFeatures.feature_names(),
            "shot": ShotCandidateFeatures.feature_names(),
        }
        Path(path).write_text(json.dumps(schema, indent=2), encoding="utf-8")
