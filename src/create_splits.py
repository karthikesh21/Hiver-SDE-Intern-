"""
Deterministic, stratified split of the 200-sample Golden Evaluation Set:
70% Development Set (140 samples) for threshold tuning and policy experimentation.
30% Held-Out Test Set (60 samples) strictly preserved for unbiased evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR, RANDOM_SEED

GOLDEN_SET_PATH = DATA_DIR / "golden_set.csv"
DEV_SET_PATH = DATA_DIR / "golden_dev_set.csv"
TEST_SET_PATH = DATA_DIR / "golden_test_set.csv"

def generate_splits(random_seed: int = RANDOM_SEED):
    print(f"Loading golden set from {GOLDEN_SET_PATH}...")
    df = pd.read_csv(GOLDEN_SET_PATH)
    
    # Stratify across joint distribution of intent and should_escalate
    df["strata"] = df["intent"] + "__" + df["should_escalate"]
    
    dev_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_seed,
        stratify=df["strata"]
    )
    
    # Drop helper column before saving
    dev_df = dev_df.drop(columns=["strata"]).sort_values("id")
    test_df = test_df.drop(columns=["strata"]).sort_values("id")
    
    dev_df.to_csv(DEV_SET_PATH, index=False)
    test_df.to_csv(TEST_SET_PATH, index=False)
    
    print(f"Created Development Set: {len(dev_df)} rows saved to {DEV_SET_PATH}")
    print(f"  AUTO_HANDLE: {sum(dev_df['should_escalate'] == 'AUTO_HANDLE')}, ESCALATE: {sum(dev_df['should_escalate'] == 'ESCALATE')}")
    print(f"Created Held-Out Test Set: {len(test_df)} rows saved to {TEST_SET_PATH}")
    print(f"  AUTO_HANDLE: {sum(test_df['should_escalate'] == 'AUTO_HANDLE')}, ESCALATE: {sum(test_df['should_escalate'] == 'ESCALATE')}")
    
    return dev_df, test_df

if __name__ == "__main__":
    generate_splits()
