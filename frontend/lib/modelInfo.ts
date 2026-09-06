/**
 * Static, descriptive copy about each model's REAL, already-documented role
 * (docs/DEVELOPMENT_ROADMAP.md, methodology/page.tsx) -- purpose/input/
 * output text, never a metric or number (those come from the live
 * `/analytics/model-performance` response only). Keying is by the exact
 * `model_name` string the backend already uses
 * (backend/app/services/analytics.py's `_DISPLAY_NAMES`).
 */
export interface ModelInfo {
  purpose: string;
  input: string;
  output: string;
}

const MODEL_INFO: Record<string, ModelInfo> = {
  intensity_persistence: {
    purpose: "Reference floor — assumes wind speed stays constant from the forecast origin.",
    input: "Current observed wind speed only.",
    output: "Wind speed forecast at +6/+12/+18/+24h.",
  },
  intensity_ridge: {
    purpose: "Linear regression baseline over engineered lag features.",
    input: "48h causal lag window (wind, pressure, position, motion).",
    output: "Wind speed forecast at +6/+12/+18/+24h.",
  },
  intensity_lightgbm: {
    purpose: "Gradient-boosted trees — the recommended production baseline for intensity.",
    input: "48h causal lag window (wind, pressure, position, motion).",
    output: "Wind speed forecast at +6/+12/+18/+24h.",
  },
  intensity_gru: {
    purpose: "Recurrent sequence model — exploratory research, did not beat LightGBM.",
    input: "48h sequence of lag features (6-hourly steps).",
    output: "Absolute wind speed forecast at +6/+12/+18/+24h.",
  },
  intensity_gru_delta: {
    purpose: "Recurrent sequence model predicting wind CHANGE — exploratory research.",
    input: "48h sequence of lag features (6-hourly steps).",
    output: "Δwind forecast at +6/+12/+18/+24h.",
  },
  track_persistence: {
    purpose: "Reference floor — assumes constant heading and speed.",
    input: "Current observed position and motion vector.",
    output: "Latitude/longitude forecast at +6/+12/+18/+24h.",
  },
  track_cliper: {
    purpose: "CLIPER-style Ridge regression — the recommended production baseline for track.",
    input: "48h causal lag window (position, motion, climatology).",
    output: "Latitude/longitude forecast at +6/+12/+18/+24h.",
  },
  track_lightgbm: {
    purpose: "Gradient-boosted trees, evaluated as an alternative track baseline.",
    input: "48h causal lag window (position, motion, climatology).",
    output: "Latitude/longitude forecast at +6/+12/+18/+24h.",
  },
  track_gru: {
    purpose: "Recurrent sequence model — exploratory research, did not beat CLIPER-style Ridge.",
    input: "48h sequence of lag features (6-hourly steps).",
    output: "Latitude/longitude forecast at +6/+12/+18/+24h.",
  },
  majority_class: {
    purpose: "Trivial reference floor — always predicts the single most common scene label.",
    input: "None (label frequency only).",
    output: "One scene_taxonomy_v1 label.",
  },
  logistic_regression: {
    purpose: "Deterministic image-statistic classifier — the recommended production baseline.",
    input: "Deterministic image statistics from the canonical Zarr store.",
    output: "One scene_taxonomy_v1 label (CDO, CurvedBand, Eye, Shear).",
  },
  resnet18: {
    purpose: "Convolutional neural network — exploratory research, did not beat Logistic Regression.",
    input: "Raw satellite image tensor.",
    output: "One scene_taxonomy_v1 label (CDO, CurvedBand, Eye, Shear).",
  },
  small_cnn: {
    purpose: "Small convolutional neural network — exploratory research, did not beat Logistic Regression.",
    input: "Raw satellite image tensor.",
    output: "One scene_taxonomy_v1 label (CDO, CurvedBand, Eye, Shear).",
  },
};

const FALLBACK: ModelInfo = {
  purpose: "Benchmarked model on the frozen test split.",
  input: "See methodology for feature details.",
  output: "See methodology for output details.",
};

export function getModelInfo(modelName: string): ModelInfo {
  return MODEL_INFO[modelName] ?? FALLBACK;
}
