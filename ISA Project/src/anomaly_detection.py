from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
import pandas as pd


def train_anomaly_model(X):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )
    model.fit(X)
    return model


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)

def predict_score(model, sensor_row: np.ndarray, feature_names: list = None) -> float:
    """
    Returns the IsolationForest decision_function score for one sample.
    """
    row = sensor_row.reshape(1, -1)
    
    if feature_names is not None:
        row = pd.DataFrame(row, columns=feature_names)
    
    return float(model.decision_function(row)[0])
