from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REGISTRY_PATH = DATA_DIR / "registry.csv"


def split_dataset(seed: int = 42):
  df = pd.read_csv(REGISTRY_PATH)

  train_df, remain_df = train_test_split(
    df,
    test_size=0.30,
    random_state=seed,
    stratify=df["label"]
  )

  val_df, test_df = train_test_split(
    remain_df,
    test_size=0.50,
    random_state=seed,
    stratify=remain_df["label"]
  )

  train_df.to_csv(DATA_DIR / "train.csv", index=False)
  val_df.to_csv(DATA_DIR / "val.csv", index=False)
  test_df.to_csv(DATA_DIR / "test.csv", index=False)

if __name__ == "__main__":
  split_dataset()