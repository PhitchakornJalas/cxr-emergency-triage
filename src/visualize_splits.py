from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sample_verification"


def plot_split_samples(split_name: str, seed: int = 42):
  """สุ่มภาพ Class 0 (1 แถว) และ Class 1 (1 แถว) จาก Split ที่ระบุ"""
  csv_path = DATA_DIR / f"{split_name}.csv"
  df = pd.read_csv(csv_path)

  sample_neg = df[df["label"] == 0].sample(n=1, random_state=seed)
  sample_pos = df[df["label"] == 1].sample(n=1, random_state=seed)
  samples_df = pd.concat([sample_neg, sample_pos], ignore_index=True)

  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  col_titles = ["1. Original CXR", "2. Label Overlay (Image + Mask)", "3. Ground Truth Mask"]

  for row_idx, (_, row) in enumerate(samples_df.iterrows()):
      img_path = PROJECT_ROOT / row["img_filepath"]
      mask_path = PROJECT_ROOT / row["seg_filepath"]
      label = int(row["label"])

      img = np.array(Image.open(img_path).convert("L"))
      mask = np.array(Image.open(mask_path).convert("L"))
      mask_bin = (mask > 0).astype(np.uint8)
      white_px = int(np.sum(mask_bin))

      axes[row_idx, 0].imshow(img, cmap="gray")
      class_text = "Class 0: Normal" if label == 0 else "Class 1: Pneumothorax"
      axes[row_idx, 0].set_ylabel(
          f"{class_text}\nID: ...{img_path.stem[-8:]}\nMask Area: {white_px:,} px",
          fontsize=10,
          fontweight="bold",
          color="#2C3E50" if label == 0 else "#C0392B"
      )

      axes[row_idx, 1].imshow(img, cmap="gray")
      if label == 1:
          masked_overlay = np.ma.masked_where(mask_bin == 0, mask_bin)
          axes[row_idx, 1].imshow(masked_overlay, cmap="autumn", alpha=0.45)

      axes[row_idx, 2].imshow(mask_bin, cmap="gray", vmin=0, vmax=1)

      if row_idx == 0:
          for col_idx, title in enumerate(col_titles):
              axes[row_idx, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=10)

      for col_idx in range(3):
          axes[row_idx, col_idx].set_xticks([])
          axes[row_idx, col_idx].set_yticks([])

  fig.suptitle(f"Dataset Verification — {split_name.upper()} Split", fontsize=14, fontweight="bold", y=0.98)
  plt.tight_layout()

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  save_path = OUTPUT_DIR / f"verify_{split_name}.png"
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"[SAVED] {save_path}")
  plt.show()


if __name__ == "__main__":
  for split in ["train", "val", "test"]:
    plot_split_samples(split_name=split, seed=42)