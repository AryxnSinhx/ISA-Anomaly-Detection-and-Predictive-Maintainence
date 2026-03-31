import pandas as pd
import glob
import os


def get_columns():
    """
    Returns column names for CMAPSS dataset
    """
    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
    return columns


def load_single_file(path):
    """
    Loads a single CMAPSS dataset file
    """
    columns = get_columns()
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )
    # Some CMAPSS files contain empty trailing columns
    df = df.dropna(axis=1)
    df.columns = columns
    return df


def load_all_training_data(folder):
    """
    Loads all CMAPSS training datasets (FD001–FD004)
    and concatenates them into one dataframe
    """

    pattern = os.path.join(folder, "train_FD*.txt")
    files = glob.glob(pattern)

    print("Searching for files using pattern:", pattern)
    print("Files detected:", files)

    datasets = []

    for file in files:
        print(f"Loading {file}")
        df = load_single_file(file)
        name = os.path.basename(file)
        # Add dataset identifier
        df["dataset"] = name
        datasets.append(df)

    if len(datasets) == 0:
        raise Exception(
            "No CMAPSS training files found.\n"
            "Make sure your folder contains:\n"
            "train_FD001.txt\n"
            "train_FD002.txt\n"
            "train_FD003.txt\n"
            "train_FD004.txt"
        )

    combined_df = pd.concat(datasets, ignore_index=True)

    print("Total rows loaded:", combined_df.shape[0])
    print("Total columns:", combined_df.shape[1])

    return combined_df