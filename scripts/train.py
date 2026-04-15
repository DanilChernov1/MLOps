import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import yaml
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data import (
    load_train,
    combine_text_features,
    TextFeatureExtractor,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> LogisticRegression:
    mc = cfg["model"]
    return LogisticRegression(
        C=mc["C"],
        max_iter=mc["max_iter"],
        solver=mc["solver"],
        random_state=mc["random_state"],
    )


def main():
    cfg = load_config()

    train_df = load_train(cfg["data"]["train_path"])

    texts = combine_text_features(train_df, cfg["features"]["text_columns"])
    labels = train_df["category_ind"].values

    fc = cfg["features"]
    mc = cfg["model"]
    extractor = TextFeatureExtractor(
        max_features=fc["max_features"],
        ngram_range=tuple(fc["ngram_range"]),
        min_df=fc["min_df"],
        sublinear_tf=fc["sublinear_tf"],
    )

    tc = cfg["training"]
    class_counts = pd.Series(labels).value_counts()
    rare_classes = set(class_counts[class_counts < 2].index)
    stratify_col = np.where(np.isin(labels, list(rare_classes)), -1, labels)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=tc["test_size"],
        random_state=tc["random_state"],
        stratify=stratify_col,
    )

    X_train = extractor.fit_transform(X_train_raw)
    X_val = extractor.transform(X_val_raw)

    mlflow.set_experiment(tc["mlflow_experiment"])

    with mlflow.start_run(run_name=tc["mlflow_run_name"]):
        mlflow.log_params(
            {
                "model_type": cfg["model"]["type"],
                "C": mc["C"],
                "max_iter": mc["max_iter"],
                "solver": mc["solver"],
                "tfidf_max_features": fc["max_features"],
                "tfidf_ngram_range": str(fc["ngram_range"]),
                "tfidf_min_df": fc["min_df"],
                "tfidf_sublinear_tf": fc["sublinear_tf"],
                "train_size": len(y_train),
                "val_size": len(y_val),
                "num_classes": len(np.unique(labels)),
            }
        )

        model = build_model(cfg)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        mlflow.log_metrics(
            {
                "val_accuracy": acc,
                "val_f1_macro": f1_macro,
                "val_f1_weighted": f1_weighted,
            }
        )

        print(f"val_accuracy:    {acc:.4f}")
        print(f"val_f1_macro:    {f1_macro:.4f}")
        print(f"val_f1_weighted: {f1_weighted:.4f}")

        extractor_path = os.path.join(MODELS_DIR, "extractor.pkl")
        model_path = os.path.join(MODELS_DIR, "model.pkl")

        with open(extractor_path, "wb") as f:
            pickle.dump(extractor, f)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(extractor_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(CONFIG_PATH)

        print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
