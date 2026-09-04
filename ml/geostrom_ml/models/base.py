"""The modularity contract for Phase 2 baseline models.

A lightweight realisation of the `BaseModel` contract in
ML_ARCHITECTURE.md §2, scoped to what Phase 2 baselines actually need:
`fit`/`predict`/`name`/`task`. ONNX export and the full Prediction/
uncertainty-field contract are deep-model concerns (Phase 5+) and are
deliberately NOT built here -- adding them now would be speculative
generality with no Phase 2 consumer, which the project's own scope-creep
guard (DEVELOPMENT_ROADMAP.md §6) warns against.

Every model implements `fit(train_df) -> None` and `predict(df) -> dict[str,
np.ndarray]` keyed by target column name, so the benchmark harness can treat
every baseline identically regardless of what's inside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaselineModel(ABC):
    task: str  # "intensity" | "track"

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> dict[str, "object"]:
        """Return {target_column_name: np.ndarray of predictions}."""
        ...
