import pandas as pd

def save_submission(ids, predictions, path="outputs/submission.csv"):
    df = pd.DataFrame(predictions, columns=[
        "task_achievement", "coherence_and_cohesion", "lexical_resource", "grammatical_range"
    ])
    df.insert(0, "ID", ids)
    df.to_csv(path, index=False)