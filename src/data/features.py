import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureExtractor:
    def __init__(
        self,
        max_features: int = 100000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
        )

    def fit_transform(self, texts: pd.Series) -> spmatrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: pd.Series) -> spmatrix:
        return self.vectorizer.transform(texts)
