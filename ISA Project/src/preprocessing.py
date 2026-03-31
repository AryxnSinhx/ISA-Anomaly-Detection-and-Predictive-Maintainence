MAX_RUL = 125

def compute_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df.apply(
        lambda row: max_cycle[row.engine_id] - row.cycle,
        axis=1
    )
    return df


def clip_rul(df, max_rul=MAX_RUL):
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df