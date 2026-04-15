import pandas as pd


def load_train(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_test(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_tree(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
