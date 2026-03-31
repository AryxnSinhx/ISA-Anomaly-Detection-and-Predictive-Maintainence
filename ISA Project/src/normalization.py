import pandas as pd


def engine_normalization(df):

    sensor_cols = [c for c in df.columns if "sensor" in c]

    normalized_engines = []

    for engine_id, engine_df in df.groupby("engine_id"):

        engine_df = engine_df.copy()

        mean = engine_df[sensor_cols].mean()
        std = engine_df[sensor_cols].std()
        std = std.replace(0, 1)
        engine_df[sensor_cols] = (engine_df[sensor_cols] - mean) / std
        normalized_engines.append(engine_df)

    df_norm = pd.concat(normalized_engines, ignore_index=True)

    return df_norm
