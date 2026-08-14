"""Learned page-pair boundary classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from document_intelligence.config.settings import get_settings
from document_intelligence.stage1.boundary_baseline import BoundaryDecision
from document_intelligence.stage1.page_features import PageFeatureBuilder, PagePairFeatures
from document_intelligence.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    train_size: int
    val_size: int
    report: dict[str, Any]
    model_path: str


class BoundaryClassifier:
    """
    Learned boundary classifier.

    Target convention:
    - label 1 = same logical document (NOT a boundary)
    - label 0 = document boundary
    """

    def __init__(self, model_type: str | None = None) -> None:
        settings = get_settings()
        self.model_type = model_type or settings.boundary_classifier_type
        self.scaler = StandardScaler()
        self.model: Any = None
        self.feature_builder = PageFeatureBuilder()
        self.threshold = 0.5

    def _create_model(self) -> Any:
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=get_settings().random_seed,
            )
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    random_state=get_settings().random_seed,
                    eval_metric="logloss",
                )
            except ImportError:
                logger.warning("XGBoost not installed, falling back to LogisticRegression")
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
    ) -> TrainingResult:
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y,
            test_size=val_size,
            random_state=get_settings().random_seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        return TrainingResult(
            train_size=len(X_train),
            val_size=len(X_val),
            report=report,
            model_path="",
        )

    def predict_proba_pairs(self, pairs: list[PagePairFeatures]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        X = np.vstack([p.features for p in pairs])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_decisions(self, pairs: list[PagePairFeatures]) -> list[BoundaryDecision]:
        probas = self.predict_proba_pairs(pairs)
        decisions: list[BoundaryDecision] = []
        for pair, prob in zip(pairs, probas):
            is_boundary = prob < self.threshold
            decisions.append(
                BoundaryDecision(
                    page_a=pair.page_a_number,
                    page_b=pair.page_b_number,
                    score=float(prob),
                    is_boundary=is_boundary,
                    method=f"learned_{self.model_type}",
                )
            )
        return decisions

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        from document_intelligence.stage1.page_features import FEATURE_NAMES

        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            return dict(zip(FEATURE_NAMES, coefs.tolist()))
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(FEATURE_NAMES, self.model.feature_importances_.tolist()))
        return {}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "threshold": self.threshold}, path)

    def load(self, path: Path | str) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.threshold = data.get("threshold", 0.5)
