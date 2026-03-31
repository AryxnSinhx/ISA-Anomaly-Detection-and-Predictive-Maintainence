import sys
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error
import tensorflow as tf

from src.feature_engineering import select_sensors, compute_health_index, SELECTED_SENSORS
from src.normalization import engine_normalization
from src.anomaly_detection import load_model as load_anomaly_model, predict_score
from src.decision_engine import decision_logic
from training.train_all_pipeline import SEQ_LENGTH          

# -------------------------------
# Paths
# -------------------------------
DATA_PATH   = PROJECT_ROOT / "data/raw"
MODEL_DIR   = PROJECT_ROOT / "models"
RESULT_PATH = MODEL_DIR    / "evaluation_results.csv"


# -------------------------------
# Load raw CMAPSS test file
# -------------------------------
def load_dataset(file_path: Path) -> pd.DataFrame:
    cols = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]   
        + [f"sensor_{i}"     for i in range(1, 22)]
    )
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1)
    df.columns = cols
    return df


# -------------------------------
# Build one sequence per engine
# -------------------------------
def create_test_sequences(df: pd.DataFrame, window: int = SEQ_LENGTH):
    feature_cols = (
        ["cycle"]
        + [c for c in df.columns if "op_setting" in c]
        + [c for c in df.columns if "sensor" in c]
        + (["health_index"] if "health_index" in df.columns else [])
    )

    X_test     = []
    engine_ids = []

    for engine in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine]
        data = engine_df[feature_cols].values

        if len(data) >= window:
            seq = data[-window:]
        else:
            padding = np.zeros((window - len(data), data.shape[1]))
            seq     = np.vstack((padding, data))

        X_test.append(seq)
        engine_ids.append(engine)

    return np.array(X_test), engine_ids


# -------------------------------
# Load ground-truth RUL file
# -------------------------------
def load_rul(file_path: Path) -> np.ndarray:
    return pd.read_csv(file_path, header=None)[0].values


# -------------------------------
# NASA asymmetric scoring function
# -------------------------------
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        diff = pred - true
        if diff < 0:
            score += np.exp(-diff / 13) - 1
        else:
            score += np.exp(diff / 10) - 1
    return score


# -------------------------------
# Main evaluation loop
# -------------------------------
datasets    = ["FD001", "FD002", "FD003", "FD004"]
results_all = []

for ds in datasets:

    print("\n===================================")
    print(f"Evaluating Dataset: {ds}")
    print("===================================")

    rul_model_path    = MODEL_DIR / f"best_rul_{ds}.h5"
    scaler_path       = MODEL_DIR / f"scaler_{ds}.pkl"
    anomaly_model_path= MODEL_DIR / f"anomaly_{ds}.pkl"
    test_file         = DATA_PATH / f"test_{ds}.txt"
    rul_file          = DATA_PATH / f"RUL_{ds}.txt"

    # --- Load artefacts ---
    print("Loading RUL model ...")
    rul_model = load_model(rul_model_path, compile=False)

    print("Loading scaler ...")
    scaler = joblib.load(scaler_path)

    print("Loading anomaly model ...")
    anomaly_model = load_anomaly_model(anomaly_model_path)

    # --- Preprocess (same steps as training) ---
    print("Loading & preprocessing test data ...")
    df_test = load_dataset(test_file)
    df_test = select_sensors(df_test)
    df_test = engine_normalization(df_test)
    df_test, _ = compute_health_index(df_test, scaler=scaler)

    # --- Sequences ---
    print("Generating sequences ...")
    X_test, engine_ids = create_test_sequences(df_test, window=SEQ_LENGTH)
    print(f"Test sequence shape: {X_test.shape}")

    # --- RUL predictions ---
    print("Predicting RUL ...")
    predictions = rul_model(X_test, training=False).numpy().flatten()

    # --- True RUL ---
    true_rul = load_rul(rul_file)

    # --- Metrics ---
    mae   = mean_absolute_error(true_rul, predictions)
    score = nasa_score(true_rul, predictions)
    print(f"MAE        : {mae:.4f}")
    print(f"NASA Score : {score:.4f}")


    sensor_cols = [c for c in df_test.columns if "sensor" in c]
    decisions   = []
    for engine, rul_pred in zip(engine_ids, predictions):
        last_row  = df_test[df_test["engine_id"] == engine][sensor_cols].values[-1]
        a_score   = predict_score(anomaly_model, last_row, feature_names=sensor_cols)  # pass names
        decision  = decision_logic(a_score, rul_pred)
        decisions.append(decision)

    # Show a breakdown of decisions for this dataset
    decision_counts = pd.Series(decisions).value_counts()
    print("\nDecision breakdown:")
    print(decision_counts.to_string())

    results_all.append({
        "dataset":    ds,
        "MAE":        round(mae, 4),
        "NASA_score": round(score, 4),
    })


# -------------------------------
# Save & print summary
# -------------------------------
results_df = pd.DataFrame(results_all)
results_df.to_csv(RESULT_PATH, index=False)

print("\n===================================")
print("Evaluation Summary")
print("=====================================")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {RESULT_PATH}")

