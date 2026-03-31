import sys
import os
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.preprocessing import compute_rul, clip_rul, MAX_RUL   
from src.feature_engineering import select_sensors, compute_health_index
from src.sequence_generator import create_sequences
from src.normalization import engine_normalization
from src.anomaly_detection import train_anomaly_model, save_model
from src.rul_model import build_model

# -------------------------------
# Paths & config
# -------------------------------
DATA_PATH = PROJECT_ROOT / "data/raw"
MODEL_DIR  = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

DATASETS    = ["FD001", "FD002", "FD003", "FD004"]
SEQ_LENGTH  = 50


# -------------------------------
# Loader
# -------------------------------
def load_dataset(file_path):
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
# Per-dataset training loop
# -------------------------------
summary_rows = []   # FIX: collect val metrics so we can print a summary at the end

for dataset in DATASETS:

    print("\n================================")
    print(f"Training model for {dataset}")
    print("================================")

    train_file = DATA_PATH / f"train_{dataset}.txt"
    df = load_dataset(train_file)

    # --- RUL ---
    df = compute_rul(df)
    df = clip_rul(df, max_rul=MAX_RUL)   # FIX: was 130 here, now uses shared MAX_RUL=125

    # --- Feature engineering ---
    df = select_sensors(df)
    df = engine_normalization(df)
    df, scaler = compute_health_index(
        df,
        save_path=MODEL_DIR / f"scaler_{dataset}.pkl"
    )

    sensor_cols = [c for c in df.columns if "sensor" in c]

    # --- Anomaly model (trained on healthy data only) ---
    healthy_df    = df[df["RUL"] > 40]
    anomaly_model = train_anomaly_model(healthy_df[sensor_cols])
    save_model(anomaly_model, MODEL_DIR / f"anomaly_{dataset}.pkl")

    # --- Engine-level train / val split (correct: no row-level leakage) ---
    engines = df["engine_id"].unique()
    train_engines, val_engines = train_test_split(
        engines, test_size=0.2, random_state=42
    )
    train_df = df[df["engine_id"].isin(train_engines)]
    val_df   = df[df["engine_id"].isin(val_engines)]

    # --- Sequences ---
    X_train, y_train = create_sequences(train_df, seq_length=SEQ_LENGTH)
    X_val,   y_val   = create_sequences(val_df,   seq_length=SEQ_LENGTH)

    print(f"Train shape : {X_train.shape}")
    print(f"Val shape   : {X_val.shape}")

    # --- Model ---
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )
    checkpoint = ModelCheckpoint(
        MODEL_DIR / f"best_rul_{dataset}.h5",
        monitor="val_loss",
        save_best_only=True
    )

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_mae  = min(history.history["val_mae"])
    print(f"\n[{dataset}] Best val_loss : {best_val_loss:.4f}")
    print(f"[{dataset}] Best val_MAE  : {best_val_mae:.4f}")

    summary_rows.append({
        "dataset":       dataset,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_mae":  round(best_val_mae,  4),
    })

    # --- Save final model ---
    model.save(MODEL_DIR / f"rul_model_{dataset}.h5")
    print(f"Model saved for {dataset}")

    #-------------------------------
    # Plot training curves for this dataset
    #-------------------------------
    plt.figure(figsize=(8,5))

    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("MAE Loss")
    plt.title("Model Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"training_loss_curve_{dataset}.png", dpi=300)
    plt.show()




# -------------------------------
# Training summary
# -------------------------------
print("\n========================================")
print("TRAINING COMPLETE — SUMMARY")
print("========================================")
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

val_mae_path = MODEL_DIR / "training_val_metrics.csv"
summary_df.to_csv(val_mae_path, index=False)
print(f"\nVal metrics saved to: {val_mae_path}")