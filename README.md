# IELTS Score Prediction

This repository provides a simple baseline for predicting IELTS writing band scores from essay text.
It trains a model to estimate the following rubric categories:

- `task_achievement`
- `coherence_and_cohesion`
- `lexical_resource`
- `grammatical_range`

## Repository structure

- `data/` &ndash; training (`df_train.csv`) and test (`df_test.csv`) datasets.
- `src/` &ndash; helper modules for preprocessing, training and prediction.
- `main.py` &ndash; example script that runs the whole pipeline.
- `outputs/` &ndash; contains a sample `submission.csv` produced by `main.py`.
- `notebooks/` &ndash; exploratory analysis notebooks.

## Getting started

Install the required packages (pandas and scikit-learn):

```bash
pip install pandas scikit-learn
```

Run the training and prediction script:

```bash
python main.py
```

This will:
1. Load and clean the training data from `data/df_train.csv`.
2. Vectorize the text using TF-IDF.
3. Train a multi-output random forest regressor and evaluate it on a validation split.
4. Predict scores for `data/df_test.csv`.
5. Write predictions to `outputs/submission.csv`.

## Data format

Each row in the training data includes a writing `prompt`, an `essay` and the four target band scores. The test data contains only the `prompt` and `essay` columns.
