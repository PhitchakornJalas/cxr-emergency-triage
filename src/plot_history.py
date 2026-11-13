import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_training_curves(model_name: str):
  # [จุดที่ 1] อัปเดตชื่อไฟล์ให้ตรงกับ Pure Segmentation History
  log_path = PROJECT_ROOT / "outputs" / "logs" / f"{model_name}_seg_history.csv"
  if not log_path.exists():
      raise FileNotFoundError(f"Training history not found at {log_path}")

  df = pd.read_csv(log_path)

  best_idx = df["val_roc_auc"].idxmax()
  best_epoch = int(df.loc[best_idx, "epoch"])
  best_roc_auc = df.loc[best_idx, "val_roc_auc"]
  best_pr_auc = df.loc[best_idx, "val_pr_auc"]
  best_dice = df.loc[best_idx, "val_pos_dice"] if "val_pos_dice" in df.columns else 0.0
  best_val_loss = df.loc[best_idx, "val_loss"]

  fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

  # --- กราฟที่ 1: Training & Validation Segmentation Loss ---
  axes[0].plot(df["epoch"], df["train_loss"], marker="o", lw=2, color="#2980B9", label="Train Loss")
  axes[0].plot(df["epoch"], df["val_loss"], marker="s", lw=2, color="#E67E22", label="Val Loss")
  axes[0].axvline(best_epoch, color="#27AE60", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
  axes[0].scatter(best_epoch, best_val_loss, color="#27AE60", s=70, zorder=5)

  axes[0].set_title(f"Training & Validation Loss — {model_name} (Seg)", fontsize=12, fontweight="bold")
  axes[0].set_xlabel("Epoch")
  # [จุดที่ 2] อัปเดตชื่อแกน Y ให้ตรงกับ Loss Function ใหม่
  axes[0].set_ylabel("Segmentation Loss (BCE + Dice)")
  axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
  axes[0].legend()
  axes[0].grid(alpha=0.3)

  # --- กราฟที่ 2: Validation ROC-AUC, PR-AUC & Positive Dice ---
  axes[1].plot(df["epoch"], df["val_roc_auc"], marker="o", lw=2, color="#8E44AD", label="Val ROC-AUC")
  axes[1].plot(df["epoch"], df["val_pr_auc"], marker="^", lw=2, color="#16A085", label="Val PR-AUC")
  # [จุดที่ 3] เพิ่มเส้น Val Pos Dice
  if "val_pos_dice" in df.columns:
    axes[1].plot(df["epoch"], df["val_pos_dice"], marker="d", lw=2, color="#D35400", label="Val Pos Dice")

  axes[1].axvline(best_epoch, color="#27AE60", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
  axes[1].scatter(best_epoch, best_roc_auc, color="#8E44AD", s=70, zorder=5)

  axes[1].annotate(
    f"Best Epoch {best_epoch}\n"
    f"ROC-AUC : {best_roc_auc:.4f}\n"
    f"PR-AUC  : {best_pr_auc:.4f}\n"
    f"Pos Dice: {best_dice:.4f}",
    xy=(best_epoch, best_roc_auc),
    xytext=(12, -35),
    textcoords="offset points",
    fontsize=9,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", fc="#EAFAF1", ec="#27AE60", lw=1),
  )

  axes[1].set_title(f"Validation Triage & Segmentation Metrics — {model_name}", fontsize=12, fontweight="bold")
  axes[1].set_xlabel("Epoch")
  axes[1].set_ylabel("Score")
  axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
  # [จุดที่ 4] ขยายแกน Y เป็น 0.0 - 1.0 เพื่อรองรับเส้น Dice Score ช่วงเริ่มต้น
  axes[1].set_ylim(0.0, 1.0)
  axes[1].legend(loc="lower right")
  axes[1].grid(alpha=0.3)

  plt.tight_layout()
  save_path = PROJECT_ROOT / "outputs" / "logs" / f"{model_name}_seg_learning_curves.png"
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"[SAVED] Learning curves -> {save_path}")
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Plot training & validation learning curves")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture name")
  args = parser.parse_args()

  plot_training_curves(args.model_name)