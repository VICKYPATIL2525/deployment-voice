# pipeline_output

Contains the trained model artifacts loaded by the API at startup. Do not modify any file here.

## Folder: `XGBoost_27032026_152209`

Named after the model and the timestamp when training completed (27 March 2026, 15:22:09).

| File | Purpose |
|---|---|
| `best_model.joblib` | Trained XGBoost classifier |
| `scaler.joblib` | RobustScaler fit on training rows — applied to every inference request |
| `label_encoder.joblib` | Maps integer class index → class name string (e.g. 3 → "normal") |
| `encoding_artifacts.joblib` | Categorical encoding maps used during training |
| `outlier_transformers.joblib` | Per-column outlier smoothing params (strategy + fitted transformer per feature) |
| `feature_names.json` | Ordered list of 1,351 acoustic feature names the model expects |
| `model_metadata.json` | Model hyperparameters, class names, feature count |
| `pipeline_state.json` | Full training pipeline log — outlier stats, feature selection steps, CV results |

## Classes predicted

`anxiety`, `bipolar`, `depression`, `normal`, `stress`, `suicidal`
