from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "segmentations"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.csv"


def build_registry():
  REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
  image_files = sorted(IMAGES_DIR.glob("*.png"))

  records = []
  for img_path in tqdm(image_files, desc="Building Registry"):
    mask_path = MASKS_DIR / img_path.name
    if not mask_path.exists():
      continue

    mask_arr = np.array(Image.open(mask_path).convert("L"))
    label = 1 if np.any(mask_arr > 0) else 0

    records.append({
      "img_filepath": str(img_path.relative_to(PROJECT_ROOT)),
      "seg_filepath": str(mask_path.relative_to(PROJECT_ROOT)),
      "label": label,
    })

  df = pd.DataFrame(records)
  df.to_csv(REGISTRY_PATH, index=False)

  print(f"\n[SAVED] {REGISTRY_PATH} ({len(df)} rows)")
  print(df["label"].value_counts())


if __name__ == "__main__":
  build_registry()