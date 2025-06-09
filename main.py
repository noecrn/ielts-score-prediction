from src.preprocess import load_and_clean_data, clean_and_concat_text, vectorize_text
from src.train import train_model, evaluate_model
from src.predict import save_submission
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMNS = [
    "task_achievement",
    "coherence_and_cohesion",
    "lexical_resource",
    "grammatical_range"
]

def main():
    # Load and clean train data
    df_train = load_and_clean_data("data/df_train.csv")
    df_train = clean_and_concat_text(df_train)
    
    # Vectorize text
    X, vectorizer = vectorize_text(df_train)
    y = df_train[TARGET_COLUMNS]

    # Train/val split and training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)

    # Predict test set
    df_test = pd.read_csv("data/df_test.csv")
    df_test["text"] = (df_test["prompt"] + " " + df_test["essay"]).str.lower()
    X_test = vectorizer.transform(df_test["text"])
    preds = model.predict(X_test)

    # Save submission
    save_submission(ids=range(1, len(preds) + 1), predictions=preds)

if __name__ == "__main__":
    main()