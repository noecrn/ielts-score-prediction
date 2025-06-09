import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer

# List of score columns
TARGET_COLUMNS = [
    "task_achievement",
    "coherence_and_cohesion",
    "lexical_resource",
    "grammatical_range"
]

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load training data and drop rows with missing target scores."""
    df = pd.read_csv(path)
    df_clean = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    return df_clean

def clean_and_concat_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, clean, and concatenate prompt + essay into a single text column."""
    def clean(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
        text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
        return text

    df["text"] = (df["prompt"] + " " + df["essay"]).apply(clean)
    return df

def vectorize_text(df: pd.DataFrame, max_features=10000):
    """TF-IDF vectorization on the 'text' column."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["text"])
    return X, vectorizer