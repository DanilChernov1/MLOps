import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data import load_train, load_test, load_tree, combine_text_features

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")) as f:
    cfg = yaml.safe_load(f)


def load_data():
    train = load_train(cfg["data"]["train_path"])
    test = load_test(cfg["data"]["test_path"])
    tree = load_tree(cfg["data"]["tree_path"])
    return train, test, tree


def print_basic_stats(train: pd.DataFrame, test: pd.DataFrame, tree: pd.DataFrame):
    print("=" * 60)
    print("BASIC STATS")
    print("=" * 60)
    print(f"Train shape:     {train.shape}")
    print(f"Test shape:      {test.shape}")
    print(f"Num categories:  {tree.shape[0]}")
    print()
    print("Train dtypes:")
    print(train.dtypes)
    print()
    print("Null values (train):")
    print(train.isnull().sum())
    print()
    null_counts = (train == "").sum()
    print("Empty string counts (train):")
    print(null_counts[null_counts > 0] if null_counts.any() else "  none")
    print()


def print_category_stats(train: pd.DataFrame, tree: pd.DataFrame):
    print("=" * 60)
    print("CATEGORY DISTRIBUTION")
    print("=" * 60)
    merged = train.merge(tree, on="category_ind")
    counts = merged["category"].value_counts()
    print(f"Total classes:        {counts.shape[0]}")
    print(f"Max samples/class:    {counts.max()}")
    print(f"Min samples/class:    {counts.min()}")
    print(f"Mean samples/class:   {counts.mean():.1f}")
    print(f"Median samples/class: {counts.median():.1f}")
    print()
    print("Top 10 categories:")
    print(counts.head(10).to_string())
    print()
    print("Bottom 10 categories:")
    print(counts.tail(10).to_string())
    print()


def plot_category_distribution(train: pd.DataFrame, tree: pd.DataFrame):
    merged = train.merge(tree, on="category_ind")
    counts = merged["category"].value_counts()

    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].hist(counts.values, bins=40, edgecolor="black", color="steelblue")
    axes[0].set_title("Distribution of samples per category")
    axes[0].set_xlabel("Number of samples")
    axes[0].set_ylabel("Number of categories")

    top20 = counts.head(20)
    short_labels = [c.split(" -> ")[-1] for c in top20.index]
    axes[1].barh(range(len(top20)), top20.values, color="steelblue")
    axes[1].set_yticks(range(len(top20)))
    axes[1].set_yticklabels(short_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Top 20 categories by sample count")
    axes[1].set_xlabel("Number of samples")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "category_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_text_lengths(train: pd.DataFrame):
    text = combine_text_features(train, cfg["features"]["text_columns"])
    lengths = text.str.split().str.len()

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(lengths.clip(upper=200), bins=50, edgecolor="black", color="coral")
    axes[0].set_title("Distribution of text length (words, clipped at 200)")
    axes[0].set_xlabel("Number of words")
    axes[0].set_ylabel("Count")

    per_field = {}
    for col in cfg["features"]["text_columns"]:
        per_field[col] = train[col].fillna("").str.split().str.len()

    df_lens = pd.DataFrame(per_field)
    df_lens.clip(upper=100).boxplot(ax=axes[1], vert=False)
    axes[1].set_title("Text length per field (words, clipped at 100)")
    axes[1].set_xlabel("Number of words")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "text_lengths.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")

    print()
    print("=" * 60)
    print("TEXT LENGTH STATS (combined)")
    print("=" * 60)
    print(lengths.describe().to_string())
    print()


def plot_top_level_categories(tree: pd.DataFrame, train: pd.DataFrame):
    tree["top_level"] = tree["category"].str.split(" -> ").str[0]
    merged = train.merge(tree[["category_ind", "top_level"]], on="category_ind")
    counts = merged["top_level"].value_counts()

    _, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="black")
    ax.set_title("Samples per top-level category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of samples")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    path = os.path.join(REPORTS_DIR, "top_level_categories.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def print_sample_rows(train: pd.DataFrame, tree: pd.DataFrame):
    print("=" * 60)
    print("SAMPLE ROWS")
    print("=" * 60)
    merged = train.merge(tree, on="category_ind")
    for _, row in merged.sample(5, random_state=42).iterrows():
        print(f"  Category: {row['category']}")
        print(f"  Name:     {row['name'][:80]}")
        print(f"  Vendor:   {row['vendor']}")
        print()


def main():
    train, test, tree = load_data()

    print_basic_stats(train, test, tree)
    print_category_stats(train, tree)
    print_sample_rows(train, tree)
    plot_category_distribution(train, tree)
    plot_text_lengths(train)
    plot_top_level_categories(tree, train)

    print()
    print("EDA complete. Reports saved to:", REPORTS_DIR)


if __name__ == "__main__":
    main()
