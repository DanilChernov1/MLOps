import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_text_features(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[col].fillna("").apply(clean_text) for col in columns]
    return pd.concat(parts, axis=1).agg(" ".join, axis=1)
