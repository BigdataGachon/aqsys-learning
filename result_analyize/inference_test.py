"""
Inference test for Air Quality LSTM model.
Loads scaler.pkl + best_model.pt + model_config.json,
then runs predictions on sample sequences from air_quality_train.csv.

Usage:
    python inference_test.py
    python inference_test.py --csv air_quality_train.csv --n 10
    python inference_test.py --manual   # predict from manually entered values
"""

import os
import json
import pickle
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# torch must be imported before sklearn to avoid OpenMP double-init segfault on macOS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="best_model.pt",    help="model checkpoint path")
parser.add_argument("--scaler", default="scaler.pkl",        help="scaler pickle path")
parser.add_argument("--config", default="model_config.json", help="model config JSON path")
parser.add_argument("--csv",    default="air_quality_train.csv", help="data CSV for sampling")
parser.add_argument("--n",      type=int, default=5,         help="number of sequences to test")
parser.add_argument("--manual", action="store_true",         help="predict from a single manually built row")
args = parser.parse_args()

# ── Device — MPS can segfault with some OpenMP builds; fall back to CPU ────────
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


# ── Model definition (must match train_model.py) ────────────────────────────
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_artifacts():
    with open(args.config) as f:
        cfg = json.load(f)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    model = AirQualityLSTM(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    model.eval()

    return cfg, scaler, model


# ── Inference on a batch of sequences ─────────────────────────────────────────
def predict(sequences_np: np.ndarray, model: AirQualityLSTM) -> np.ndarray:
    """sequences_np: (N, seq_len, n_features) already scaled. Returns (N, 2)."""
    with torch.no_grad():
        x = torch.tensor(sequences_np, dtype=torch.float32).to(DEVICE)
        preds = model(x)
    return preds.cpu().numpy()


# ── Build sequences from CSV ───────────────────────────────────────────────────
def build_sequences_from_csv(cfg, scaler):
    df = pd.read_csv(args.csv)
    feature_cols = cfg["features"]
    target_cols  = cfg["targets"]
    seq_len      = cfg["seq_len"]

    df = df.sort_values(["district", "date", "hour"]).reset_index(drop=True)
    feature_data = scaler.transform(df[feature_cols].values.astype(np.float32))
    target_data  = df[target_cols].values.astype(np.float32)
    districts    = df["district"].values

    sequences, targets, meta = [], [], []
    n = len(df)
    for i in range(seq_len, n):
        if districts[i - seq_len] != districts[i]:
            continue
        sequences.append(feature_data[i - seq_len : i])
        targets.append(target_data[i])
        meta.append({
            "district": districts[i],
            "date":     df["date"].iloc[i],
            "hour":     df["hour"].iloc[i],
        })

    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets,   dtype=np.float32),
        meta,
    )


# ── Manual single-step example ─────────────────────────────────────────────────
def build_manual_sequence(cfg, scaler):
    """
    Constructs a synthetic 24-hour sequence with constant conditions
    as a sanity-check input. Edit the values here to probe the model.
    """
    seq_len     = cfg["seq_len"]
    feature_cols = cfg["features"]

    # Representative Seoul winter hour values — adjust as needed
    row = {
        "hour":          9,
        "month":         1,
        "day_of_week":   0,
        "season":        0,
        "temperature":  -2.0,
        "humidity":      55.0,
        "dew_point":    -10.0,
        "wind_speed":    2.5,
        "wind_direction": 90.0,
        "pressure":     1020.0,
        "no2":           0.035,
        "so2":           0.006,
        "co":            0.8,
        "o3":            0.012,
        "pm10":          45.0,
        "pm25":          22.0,
    }

    base = np.array([[row[c] for c in feature_cols]], dtype=np.float32)
    sequence = np.repeat(base, seq_len, axis=0)   # (seq_len, n_features)
    sequence_scaled = scaler.transform(sequence)
    return sequence_scaled[np.newaxis, ...]        # (1, seq_len, n_features)


# ── Pretty print ───────────────────────────────────────────────────────────────
def print_results(preds, targets=None, meta=None, n_show=None):
    n = len(preds) if n_show is None else min(n_show, len(preds))
    width = 68

    print("\n" + "═" * width)
    print("  INFERENCE RESULTS")
    print("═" * width)
    header = f"{'#':>4}  {'District':<8}  {'Date':>10}  {'Hr':>2}  "
    header += f"{'PM10 pred':>10}  {'PM10 actual':>11}"
    header += f"  {'PM25 pred':>10}  {'PM25 actual':>11}"
    if targets is not None:
        print(header)
    else:
        print(f"{'#':>4}  {'Info':<22}  {'PM10 pred':>10}  {'PM25 pred':>10}")
    print("─" * width)

    for i in range(n):
        pm10_p, pm25_p = preds[i]
        if targets is not None and meta is not None:
            pm10_a, pm25_a = targets[i]
            m = meta[i]
            err10 = pm10_p - pm10_a
            err25 = pm25_p - pm25_a
            print(
                f"  {i+1:>2}  {m['district']:<8}  {m['date']:>10}  {m['hour']:>2}h  "
                f"  {pm10_p:>8.2f}    {pm10_a:>8.2f} ({err10:+.2f})"
                f"  {pm25_p:>8.2f}    {pm25_a:>8.2f} ({err25:+.2f})"
            )
        else:
            print(f"  {i+1:>2}  {'manual input':<22}  {pm10_p:>10.2f}  {pm25_p:>10.2f}")

    print("─" * width)

    if targets is not None:
        mae10 = float(np.abs(preds[:n, 0] - targets[:n, 0]).mean())
        mae25 = float(np.abs(preds[:n, 1] - targets[:n, 1]).mean())
        print(f"  MAE on shown samples — PM10: {mae10:.3f} μg/m³   PM25: {mae25:.3f} μg/m³")

    print("═" * width)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Loading artifacts ...")
    cfg, scaler, model = load_artifacts()

    print(f"  model  : {args.model}")
    sklearn_ver = getattr(scaler, "_sklearn_version", "unknown")
    print(f"  scaler : {args.scaler}  (sklearn {sklearn_ver})")
    print(f"  config : seq_len={cfg['seq_len']}, features={cfg['input_size']}, "
          f"hidden={cfg['hidden_size']}×{cfg['num_layers']}")
    print(f"  Reported test MAE — PM10: {cfg['test_mae_pm10']:.3f}  PM25: {cfg['test_mae_pm25']:.3f}")

    if args.manual:
        seq = build_manual_sequence(cfg, scaler)
        preds = predict(seq, model)
        print_results(preds, targets=None, meta=None)
        return

    # Sample from CSV
    print(f"\nBuilding sequences from {args.csv} ...")
    sequences, targets, meta = build_sequences_from_csv(cfg, scaler)
    print(f"  Total sequences: {len(sequences):,}")

    # Sample evenly across the dataset so we see diverse conditions
    indices = np.linspace(0, len(sequences) - 1, args.n, dtype=int)
    sample_seq = sequences[indices]
    sample_tgt = targets[indices]
    sample_meta = [meta[i] for i in indices]

    preds = predict(sample_seq, model)
    print_results(preds, targets=sample_tgt, meta=sample_meta)

    # Full-dataset MAE for reference
    print("\nRunning full-dataset inference ...")
    batch = 2048
    all_preds = []
    for start in range(0, len(sequences), batch):
        all_preds.append(predict(sequences[start : start + batch], model))
    all_preds = np.concatenate(all_preds)
    mae10 = float(np.abs(all_preds[:, 0] - targets[:, 0]).mean())
    mae25 = float(np.abs(all_preds[:, 1] - targets[:, 1]).mean())
    rmse10 = float(np.sqrt(((all_preds[:, 0] - targets[:, 0]) ** 2).mean()))
    rmse25 = float(np.sqrt(((all_preds[:, 1] - targets[:, 1]) ** 2).mean()))
    print(f"  PM10  MAE={mae10:.3f}  RMSE={rmse10:.3f}  μg/m³")
    print(f"  PM25  MAE={mae25:.3f}  RMSE={rmse25:.3f}  μg/m³")


if __name__ == "__main__":
    main()
