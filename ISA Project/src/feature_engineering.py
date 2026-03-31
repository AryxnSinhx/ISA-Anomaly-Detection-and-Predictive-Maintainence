from sklearn.preprocessing import MinMaxScaler
import joblib
import os

SELECTED_SENSORS = [
    'sensor_2',  'sensor_3',  'sensor_4',
    'sensor_7',  'sensor_8',  'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21'
]


def select_sensors(df):
    cols = ["engine_id", "cycle"] + SELECTED_SENSORS

    if "RUL" in df.columns:
        cols.append("RUL")

    return df[cols]


def compute_health_index(df, scaler=None, save_path=None):
    sensors = [c for c in SELECTED_SENSORS if c in df.columns]

    if not sensors:
        raise ValueError(
            "No selected sensors found in dataframe. "
            "Ensure select_sensors() has been called or the dataframe "
            "contains columns from SELECTED_SENSORS."
        )

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[sensors])

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(scaler, save_path)

    else:
        scaled = scaler.transform(df[sensors])

    df = df.copy()
    df["health_index"] = 1 - scaled.mean(axis=1)

    return df, scaler
