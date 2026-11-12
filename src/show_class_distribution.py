from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "class_distribution.png"


def show_class_distribution():
  splits = ["train", "val", "test"]
  summary_rows = []

  for split_name in splits:
    csv_path = DATA_DIR / f"{split_name}.csv"
    df = pd.read_csv(csv_path)

    total = len(df)
    neg = int((df["label"] == 0).sum())
    pos = int((df["label"] == 1).sum())
    pos_ratio = (pos / total) * 100

    summary_rows.append({
      "Split": split_name.capitalize(),
      "Total": total,
      "Normal (0)": neg,
      "Pneumothorax (1)": pos,
      "Positive (%)": round(pos_ratio, 2)
    })

  summary_df = pd.DataFrame(summary_rows)

  print("\n" + "=" * 58)
  print("DATASET CLASS DISTRIBUTION SUMMARY")
  print("=" * 58)
  print(summary_df.to_string(index=False))
  print("=" * 58)

  x = np.arange(len(summary_df))
  width = 0.35

  fig, ax = plt.subplots(figsize=(8, 5))
  rects1 = ax.bar(x - width / 2, summary_df["Normal (0)"], width, label="Normal (0)", color="#4A90E2")
  rects2 = ax.bar(x + width / 2, summary_df["Pneumothorax (1)"], width, label="Pneumothorax (1)", color="#E94E77")

  ax.set_title("Class Distribution Across Train / Val / Test Splits", fontsize=13, fontweight="bold", pad=12)
  ax.set_ylabel("Number of Images", fontsize=11)
  ax.set_xticks(x)
  ax.set_xticklabels(summary_df["Split"], fontsize=11)
  ax.legend(frameon=True)
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  for idx, (r1, r2) in enumerate(zip(rects1, rects2)):
    h1, h2 = r1.get_height(), r2.get_height()
    pct = summary_df.loc[idx, "Positive (%)"]

    ax.annotate(f"{h1:,}",
                xy=(r1.get_x() + r1.get_width() / 2, h1),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

    ax.annotate(f"{h2:,}\n({pct}%)",
                xy=(r2.get_x() + r2.get_width() / 2, h2),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#C0392B")

  ax.set_ylim(0, summary_df["Normal (0)"].max() * 1.15)

  plt.tight_layout()
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(OUTPUT_PATH, dpi=150)
  print(f"[SAVED] Plot saved to: {OUTPUT_PATH}")
  plt.show()

if __name__ == "__main__":
  show_class_distribution()