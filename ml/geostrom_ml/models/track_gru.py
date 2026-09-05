"""Phase 8: GRU sequence model for track prediction.

Implements the Tier-2 "MVP" model `docs/ML_ARCHITECTURE.md` §7.4 already
specified (locked in Phase 0, not invented this phase): **"GRU encoder ->
dense multi-output head producing 8 displacement values"**, trained on the
exact same causal `(L=8, F=20)` window Phase 2/Phase 7 already build
(`ml/geostrom_ml/features/engineering.py` -- reused unmodified) and evaluated
with the same `track_point_metrics()`/`evaluate_track_model()` functions
Phase 2's benchmark harness uses (`ml/geostrom_ml/evaluation/{metrics,
benchmark}.py` -- reused unmodified, zero changes, not reimplemented).

Per §7.2's own words ("Same window as intensity; the two models share the
feature pipeline"), the low-level sequence-reshaping and determinism helpers
are IMPORTED from `ml.geostrom_ml.models.intensity_gru` rather than
duplicated -- both modules consume the identical `x__<feat>__lag{k}` window
format, so a second, subtly-divergent copy of the same reindexing logic would
be the exact kind of duplicate implementation the project's conventions
warn against. This is a closer, same-subsystem reuse than Phase 6's decision
NOT to import across the image-classification package, which is a genuinely
different data modality.

Conforms to the same `BaselineModel` contract (`fit`/`predict`/`name`/`task`)
Phase 2's track baselines use (`ml/geostrom_ml/models/track_baselines.py`),
so `ml/geostrom_ml/evaluation/benchmark.py::evaluate_track_model` can
evaluate this model with ZERO changes to that file. `predict()` returns raw,
UNWEIGHTED (dlat, dlon) degree displacements -- exactly like
`CliperTrack`/`LightGBMTrack` -- so absolute-position reconstruction and all
downstream metrics are computed identically for every model. The
cos(latitude) longitude weighting described in ML_ARCHITECTURE.md §7.2 is a
TRAINING-loss-only construction (see `CosLatWeightedHuberLoss` below); it
never appears in a prediction or a stored metric.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
from ml.geostrom_ml.features.geo import displace, haversine_km  # noqa: E402
from ml.geostrom_ml.models.base import BaselineModel  # noqa: E402
from ml.geostrom_ml.models.intensity_gru import (  # noqa: E402
    N_FEATURES, RANDOM_SEED, _resolve_device, reshape_to_sequence, set_deterministic,
)
from ml.geostrom_ml.models.track_baselines import dlat_col, dlon_col  # noqa: E402

FEATURE_COLS = flattened_feature_columns()


def lat_future_col(h: int) -> str:
    return f"y_lat_future_{h}h"


def lon_future_col(h: int) -> str:
    return f"y_lon_future_{h}h"


class CosLatWeightedHuberLoss(nn.Module):
    """Huber loss on (Δlat, Δlon) displacements, with the longitude
    component weighted by cos(reference latitude), per
    `docs/ML_ARCHITECTURE.md` §7.2: **"Loss: Huber on scaled displacements,
    with longitude displacement weighted by cos(latitude) ... this detail
    matters. One degree of longitude is ~111 km at the equator but ~55 km at
    60N. Unweighted, the loss over-penalises high-latitude longitude error
    ... The cos-latitude weight makes the loss approximate true distance."**

    Exact formulation, per sample i and horizon h:
        w_i          = cos(radians(ref_lat_i))          -- fixed per window,
                        computed from the ALREADY-OBSERVED reference
                        position (the last input timestep), never a future
                        value, so this introduces no leakage.
        e_lat_{i,h}  = pred_dlat_{i,h}  - true_dlat_{i,h}
        e_lon_{i,h}  = w_i * (pred_dlon_{i,h} - true_dlon_{i,h})
        loss = mean_{i,h}[ Huber(e_lat_{i,h}) ] + mean_{i,h}[ Huber(e_lon_{i,h}) ]

    "Scaled displacements": the two model inputs this loss consumes are
    already train-split-only StandardScaler-normalised (matching
    `IntensityGRU`'s established convention); the (dlat, dlon) TARGETS
    themselves are left in raw degrees, unlike the input features -- lat and
    lon displacements are the same physical unit (degrees) and, once the
    longitude term is cos-weighted, are already directly comparable, unlike
    e.g. combining knots and kilometres, where standardisation would be
    load-bearing. This is a deliberate, documented choice, not an omission
    (per the task's explicit "do not invent a different geographic loss
    without documenting why").
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self._huber = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(self, pred: torch.Tensor, true: torch.Tensor,
                cos_ref_lat: torch.Tensor) -> torch.Tensor:
        # pred, true: (batch, n_horizons, 2) -- [..., 0]=dlat, [..., 1]=dlon
        lat_loss = self._huber(pred[..., 0], true[..., 0])
        w = cos_ref_lat.unsqueeze(-1)  # (batch, 1) broadcasts over horizons
        lon_loss = self._huber(w * pred[..., 1], w * true[..., 1])
        return lat_loss + lon_loss


class _TrackGRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, n_horizons: int):
        super().__init__()
        self.n_horizons = n_horizons
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_horizons * 2),  # 8 outputs: (dlat,dlon) x 4 horizons
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)                  # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]                  # final layer's hidden state
        out = self.head(last_hidden)           # (batch, n_horizons*2)
        return out.view(-1, self.n_horizons, 2)  # (batch, n_horizons, 2)


@dataclass
class TrackGRUConfig:
    """Every training knob, explicit. Defaults mirror Phase 7's
    `GRUIntensityConfig` (same dataset scale, same compute budget, same
    roadmap-authorised "1-2 layers, hidden 64-128" range in
    `ML_ARCHITECTURE.md` §7.4/§9) -- not imported, since track and intensity
    are independently versioned models with their own config identity, but
    deliberately kept at the same conservative point in the range rather
    than re-opening a hyperparameter search this phase does not authorise."""
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    huber_delta: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping_patience: int = 15   # epochs without val-track-km improvement
    early_stopping_min_delta: float = 0.0
    seed: int = RANDOM_SEED
    device: str = "auto"


class TrackGRU(BaselineModel):
    """GRU (1-2 layers) -> dropout -> dense multi-output head producing 8
    values: (Δlat, Δlon) at +6/+12/+18/+24h, per §7.1's flow diagram. One
    shared model produces all horizons simultaneously (matching
    `IntensityGRU`'s multi-output-head design and §7.4's "dense multi-output
    head" spec), unlike the one-model-per-horizon-per-axis pattern the
    tabular baselines (`CliperTrack`, `LightGBMTrack`) use."""

    task = "track"

    def __init__(self, horizons_h: tuple[int, ...] = HORIZONS_H,
                 config: TrackGRUConfig | None = None):
        super().__init__(name="track_gru_v1")
        self.horizons_h = list(horizons_h)
        self.n_horizons = len(self.horizons_h)
        self.config = config or TrackGRUConfig()
        self._model: _TrackGRURegressor | None = None
        self._scaler: StandardScaler | None = None
        self._col_medians: pd.Series | None = None
        self.history: list[dict] = []
        self.best_epoch: int = -1

    def _prep_X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].copy()
        if self._col_medians is not None:
            X = X.fillna(self._col_medians)  # same train-only-median rule as CliperTrack
        X_scaled = self._scaler.transform(X) if self._scaler is not None else X.to_numpy()
        scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=df.index)
        return reshape_to_sequence(scaled_df, L=L_STEPS)

    def _prep_y(self, df: pd.DataFrame) -> np.ndarray:
        y = np.empty((len(df), self.n_horizons, 2), dtype=np.float32)
        for i, h in enumerate(self.horizons_h):
            y[:, i, 0] = df[dlat_col(h)].to_numpy(dtype=np.float32)
            y[:, i, 1] = df[dlon_col(h)].to_numpy(dtype=np.float32)
        return y

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        set_deterministic(self.config.seed)
        device = _resolve_device(self.config.device)

        self._col_medians = train_df[FEATURE_COLS].median()
        X_train_raw = train_df[FEATURE_COLS].fillna(self._col_medians)
        self._scaler = StandardScaler().fit(X_train_raw)  # train-split-only, §7.2 shared convention

        X_train = self._prep_X(train_df)
        y_train = self._prep_y(train_df)
        w_train = np.cos(np.radians(train_df["ref_lat"].to_numpy(dtype=np.float32)))

        has_val = val_df is not None and len(val_df) > 0
        if has_val:
            X_val = self._prep_X(val_df)
            val_ref_lat = val_df["ref_lat"].to_numpy(dtype=np.float64)
            val_ref_lon = val_df["ref_lon"].to_numpy(dtype=np.float64)
            val_true_lat = {h: val_df[lat_future_col(h)].to_numpy(dtype=np.float64)
                            for h in self.horizons_h}
            val_true_lon = {h: val_df[lon_future_col(h)].to_numpy(dtype=np.float64)
                            for h in self.horizons_h}

        model = _TrackGRURegressor(N_FEATURES, self.config.hidden_size, self.config.num_layers,
                                   self.config.dropout, self.n_horizons).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        criterion = CosLatWeightedHuberLoss(delta=self.config.huber_delta)  # §7.2

        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train),
                          torch.from_numpy(w_train)),
            batch_size=self.config.batch_size, shuffle=True, generator=generator,
        )

        best_val_km = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(self.config.max_epochs):
            model.train()
            epoch_loss, n_seen = 0.0, 0
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb, wb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = epoch_loss / max(n_seen, 1)

            record = {"epoch": epoch, "train_weighted_huber_loss": train_loss}
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
                # Primary, physically-meaningful selection metric: mean
                # great-circle error (km), averaged across all 4 horizons --
                # matches ML_ARCHITECTURE.md §7.2's own primary metric,
                # reconstructed via the same displace()/haversine_km()
                # functions the evaluation harness uses (never re-derived).
                per_horizon_km = []
                for i, h in enumerate(self.horizons_h):
                    pred_lat, pred_lon = displace(
                        val_ref_lat, val_ref_lon, val_pred[:, i, 0], val_pred[:, i, 1],
                    )
                    err_km = haversine_km(val_true_lat[h], val_true_lon[h], pred_lat, pred_lon)
                    per_horizon_km.append(np.mean(err_km))
                val_track_km = float(np.mean(per_horizon_km))
                record["val_mean_track_error_km"] = val_track_km

                improved = val_track_km < best_val_km - self.config.early_stopping_min_delta
                if improved:
                    best_val_km, self.best_epoch = val_track_km, epoch
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

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Returns raw, UNWEIGHTED (dlat, dlon) degree displacements per
        horizon -- exactly like `CliperTrack`/`LightGBMTrack` -- so this
        model plugs directly into
        `ml/geostrom_ml/evaluation/benchmark.py::evaluate_track_model`
        exactly like every Phase 2 baseline, no changes to that file."""
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        X = self._prep_X(df)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X).to(self._device)).cpu().numpy()
        out = {}
        for i, h in enumerate(self.horizons_h):
            out[dlat_col(h)] = pred[:, i, 0].astype(np.float64)
            out[dlon_col(h)] = pred[:, i, 1].astype(np.float64)
        return out
