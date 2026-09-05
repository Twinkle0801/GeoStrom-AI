"""Phase 7: GRU sequence model for intensity prediction.

Implements the Tier-2 "MVP" model `docs/ML_ARCHITECTURE.md` §6.5 already
specified (locked in Phase 0, not invented this phase): **"GRU (1-2 layers,
hidden 64-128) -> dropout -> dense multi-output head"**, Huber loss (§6.2),
trained on the exact same causal `(L=8, F=20)` window features Phase 2 built
(`ml/geostrom_ml/features/engineering.py` -- reused unmodified) and evaluated
with the same `intensity_metrics()`/`skill_vs_baseline()` functions Phase 2's
benchmark harness uses (`ml/geostrom_ml/evaluation/{metrics,benchmark}.py` --
reused unmodified, not reimplemented).

Conforms to the same `BaselineModel` contract (`fit`/`predict`/`name`/`task`)
Phase 2's baselines use, so `ml/geostrom_ml/evaluation/benchmark.py::
evaluate_intensity_model` can evaluate this model with ZERO changes to that
file -- `fit()` gains an additional optional `val_df` keyword (needed for
early stopping; Python subclassing does not require an identical signature)
but is still callable as `fit(train_df)` exactly like every Phase 2 baseline.

Per §6.4 ("DECISION: train both, report both"), this module supports BOTH
target variants:
  - `target_mode="absolute"` -- predicts wind directly (the deliverable).
  - `target_mode="delta"` -- predicts intensity CHANGE (Delta-wind, the
    diagnostic that exposes whether real skill exists beyond persistence).
    `predict()` always returns ABSOLUTE wind (reconstructed as
    `ref_wind + predicted_delta`, exactly as `docs/ML_ARCHITECTURE.md` §6.4
    specifies for serving) so both variants plug into the same evaluation
    path; `predict_delta()` returns the raw, unreconstructed delta
    predictions for delta-scale diagnostics (e.g. RI recall).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.geostrom_ml.features.engineering import (  # noqa: E402
    HORIZONS_H, L_STEPS, PER_TIMESTEP_FEATURES, flattened_feature_columns,
)
from ml.geostrom_ml.models.base import BaselineModel  # noqa: E402

FEATURE_COLS = flattened_feature_columns()
N_FEATURES = len(PER_TIMESTEP_FEATURES)
RANDOM_SEED = 42  # matches ml/geostrom_ml/splits/split.py::RANDOM_SEED


def target_col(h: int) -> str:
    return f"y_wind_abs_{h}h"


def delta_target_col(h: int) -> str:
    return f"y_wind_delta_{h}h"


def set_deterministic(seed: int = RANDOM_SEED) -> None:
    """Same discipline as Phase 6's `classification/deep/training.py::
    set_deterministic` (seed before any stochastic op, incl. model
    construction) -- reimplemented here, not imported, since this module has
    no dependency on the image-classification package and importing across
    unrelated ML subsystems for three lines of code would be the wrong kind
    of coupling. Documented limitation, same as Phase 6: CUDA convolution/
    kernel selection is not guaranteed bit-exact across GPU driver/cuDNN
    versions in general (not exercised here in practice -- see §15 of the
    Phase 7 doc; this GRU is tiny enough that CPU training is already fast,
    see the compute-budget note in `docs/ML_ARCHITECTURE.md` §9)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reshape_to_sequence(df: pd.DataFrame, L: int = L_STEPS) -> np.ndarray:
    """Reshape the flattened `x__<feat>__lag{k}` columns (Phase 2's window
    format, lag0=t_ref=most recent ... lag(L-1)=oldest) into a chronological
    `(N, L, F)` tensor for a recurrent model: sequence index 0 = oldest
    (lag L-1), sequence index L-1 = most recent (lag0) -- so the GRU reads
    the storm's history in the order it actually happened.

    Pure reindexing of already-computed, already-causality-verified Phase 2
    columns -- no new feature is computed here, so this cannot introduce any
    leakage that build_sequence_windows() did not already guard against.
    """
    n = len(df)
    out = np.empty((n, L, N_FEATURES), dtype=np.float32)
    for k in range(L):
        seq_idx = L - 1 - k  # lag k -> chronological position (lag0 -> last)
        cols = [f"x__{feat}__lag{k}" for feat in PER_TIMESTEP_FEATURES]
        out[:, seq_idx, :] = df[cols].to_numpy(dtype=np.float32)
    return out


class _GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, n_horizons: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]          # final layer's hidden state
        return self.head(last_hidden)  # (batch, n_horizons)


@dataclass
class GRUIntensityConfig:
    """Every training knob, explicit -- per the task's "explicit
    configuration" requirement. Defaults are the specific point chosen
    inside `docs/ML_ARCHITECTURE.md` §6.5's stated range ("1-2 layers,
    hidden 64-128"): 1 layer (simplest choice that satisfies the range,
    appropriate for an 8-step sequence per §6.5's own GRU-vs-LSTM
    rationale), hidden 64 (low end of the range -- ~6,100 train windows is
    modest; a smaller network is more conservative against overfitting,
    consistent with Risk #13 in the roadmap's risk register)."""
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256          # matches docs/ML_ARCHITECTURE.md §9's GRU compute-budget row
    max_epochs: int = 200
    early_stopping_patience: int = 15   # epochs without val-MAE improvement
    early_stopping_min_delta: float = 0.0
    seed: int = RANDOM_SEED
    device: str = "auto"


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class IntensityGRU(BaselineModel):
    """GRU (1-2 layers) -> dropout -> dense multi-output head, one model
    producing all `HORIZONS_H` outputs simultaneously (a genuine multi-output
    head, per §6.1's flow diagram, rather than the one-model-per-horizon
    pattern the tabular baselines use -- a sequence model's hidden state
    already summarises the whole window, so one shared head over it is the
    natural fit; §6.5 specifies "dense multi-output head" for exactly this
    reason)."""

    task = "intensity"

    def __init__(self, target_mode: str = "absolute",
                 horizons_h: tuple[int, ...] = HORIZONS_H,
                 config: GRUIntensityConfig | None = None):
        if target_mode not in ("absolute", "delta"):
            raise ValueError(f"target_mode must be 'absolute' or 'delta', got {target_mode!r}")
        name = "intensity_gru_v1" if target_mode == "absolute" else "intensity_gru_delta_v1"
        super().__init__(name=name)
        self.target_mode = target_mode
        self.horizons_h = list(horizons_h)
        self.config = config or GRUIntensityConfig()
        self._model: _GRURegressor | None = None
        self._scaler: StandardScaler | None = None
        self._col_medians: pd.Series | None = None
        self.history: list[dict] = []
        self.best_epoch: int = -1

    def _target_cols(self) -> list[str]:
        fn = target_col if self.target_mode == "absolute" else delta_target_col
        return [fn(h) for h in self.horizons_h]

    def _prep_X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].copy()
        if self._col_medians is not None:
            X = X.fillna(self._col_medians)  # same train-only-median rule as RidgeIntensity
        X_scaled = self._scaler.transform(X) if self._scaler is not None else X.to_numpy()
        scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=df.index)
        return reshape_to_sequence(scaled_df, L=L_STEPS)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        set_deterministic(self.config.seed)
        device = _resolve_device(self.config.device)

        self._col_medians = train_df[FEATURE_COLS].median()
        X_train_raw = train_df[FEATURE_COLS].fillna(self._col_medians)
        self._scaler = StandardScaler().fit(X_train_raw)  # train-split-only, per §6.1 "SCALING"

        X_train = self._prep_X(train_df)
        y_train = train_df[self._target_cols()].to_numpy(dtype=np.float32)

        has_val = val_df is not None and len(val_df) > 0
        if has_val:
            X_val = self._prep_X(val_df)
            y_val = val_df[self._target_cols()].to_numpy(dtype=np.float32)
            val_ref_wind = val_df["ref_wind"].to_numpy(dtype=np.float64)
            val_true_abs = val_df[[target_col(h) for h in self.horizons_h]].to_numpy(dtype=np.float64)

        model = _GRURegressor(N_FEATURES, self.config.hidden_size, self.config.num_layers,
                              self.config.dropout, len(self.horizons_h)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        criterion = nn.HuberLoss()  # §6.2 "Loss: Huber (smooth L1)"

        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.config.batch_size, shuffle=True, generator=generator,
        )

        best_val_mae = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(self.config.max_epochs):
            model.train()
            epoch_loss, n_seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = epoch_loss / max(n_seen, 1)

            record = {"epoch": epoch, "train_huber_loss": train_loss}
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
                if self.target_mode == "delta":
                    val_pred_abs = val_ref_wind[:, None] + val_pred.astype(np.float64)
                else:
                    val_pred_abs = val_pred.astype(np.float64)
                val_mae = float(np.mean(np.abs(val_true_abs - val_pred_abs)))
                record["val_mae_kt"] = val_mae

                improved = val_mae < best_val_mae - self.config.early_stopping_min_delta
                if improved:
                    best_val_mae, self.best_epoch = val_mae, epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            self.history.append(record)

            if has_val and epochs_without_improvement >= self.config.early_stopping_patience:
                break

        if has_val and best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._device = device

    def _predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        X = self._prep_X(df)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X).to(self._device)).cpu().numpy()
        return pred.astype(np.float64)

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Always returns ABSOLUTE wind per horizon (reconstructed from
        `ref_wind + predicted_delta` for the delta-target variant, per
        docs/ML_ARCHITECTURE.md §6.4) -- so this model plugs directly into
        `ml/geostrom_ml/evaluation/benchmark.py::evaluate_intensity_model`
        exactly like every Phase 2 baseline, no changes to that file."""
        raw = self._predict_raw(df)
        if self.target_mode == "delta":
            ref_wind = df["ref_wind"].to_numpy(dtype=np.float64)
            raw = ref_wind[:, None] + raw
        return {target_col(h): raw[:, i] for i, h in enumerate(self.horizons_h)}

    def predict_delta(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Raw (unreconstructed) predictions on whatever scale this model
        was trained on -- absolute wind if `target_mode='absolute'`, else
        true Delta-wind. Used only for delta-scale diagnostics (RI recall);
        `predict()` above is the contract every evaluation path should use."""
        raw = self._predict_raw(df)
        fn = target_col if self.target_mode == "absolute" else delta_target_col
        return {fn(h): raw[:, i] for i, h in enumerate(self.horizons_h)}
