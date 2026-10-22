"""Data ingestion and train-test split."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE

KAGGLE_DATASET = "rmisra/news-headlines-dataset-for-sarcasm-detection"
FILE_NAME = "Sarcasm_Headlines_Dataset_v2.json"


def load_data(path=None):
    """Load the headlines dataset as a DataFrame with columns `is_sarcastic` and `headline`.

    If `path` is not given, the dataset is downloaded from Kaggle with `kagglehub`.
    `path` can be a folder containing the JSON file or the JSON file itself.
    """
    if path is None:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)

    full_path = path if path.endswith(".json") else os.path.join(path, FILE_NAME)
    df = pd.read_json(full_path, lines=True)

    # the article link is not used as a feature
    return df.drop("article_link", axis=1)


def split_data(df, test_size=0.2):
    """80/20 train-test split of headlines (x) and labels (y)."""
    x = df["headline"]
    y = df["is_sarcastic"]
    return train_test_split(x, y, test_size=test_size, random_state=RANDOM_STATE)
