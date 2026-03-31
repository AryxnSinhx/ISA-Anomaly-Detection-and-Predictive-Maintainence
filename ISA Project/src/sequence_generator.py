import numpy as np


def create_sequences(df, seq_length=50):

    sequences = []
    labels = []

    engines = df["engine_id"].unique()
    
    features = (
        ["cycle"]
        + [c for c in df.columns if "op_setting" in c]
        + [c for c in df.columns if "sensor" in c]
        + (["health_index"] if "health_index" in df.columns else [])
    )

    for engine in engines:

        engine_df = df[df["engine_id"] == engine]

        data = engine_df[features + ["RUL"]].values

        for i in range(len(data) - seq_length):

            seq   = data[i : i + seq_length, :-1]
            label = data[i + seq_length, -1]

            sequences.append(seq)
            labels.append(label)

    return np.array(sequences), np.array(labels)
