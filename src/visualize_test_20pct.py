import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import get_transforms
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_calibrated_threshold(model_name: str, default: float = 0.12) -> float:
  metrics_path = PROJECT_ROOT / "outputs" / "evaluations" / f"{model_name}_seg_metrics.json"
  if metrics_path.exists():
    with open(metrics_path, "r") as f:
      return float(json.load(f)["clinical_triage"]["threshold"])
  return default


@torch.no_grad()
def main(args):
  device = get_device()
  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.model_name}_seg_best.pth"
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
  img_size = checkpoint.get("img_size", 256)
  threshold = load_calibrated_threshold(args.model_name)

  model = build_model(model_name=args.model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")
  sample_df = (
    test_df.groupby("label", group_keys=False)
    .sample(frac=args.frac, random_state=args.seed)
    .sample(frac=1.0, random_state=args.seed)
    .reset_index(drop=True)
  )

  total_samples = len(sample_df)
  per_page = 20  # 4 แถว x 5 คอลัมน์ ต่อ 1 รูปภาพ
  num_pages = int(np.ceil(total_samples / per_page))
  print(f"[INFO] Visualizing {total_samples}/{len(test_df)} test images ({args.frac*100:.0f}%) across {num_pages} pages...")

  _, eval_tf = get_transforms(img_size=img_size)
  out_dir = PROJECT_ROOT / "outputs" / "evaluations" / f"{args.model_name}_test_20pct_grid"
  out_dir.mkdir(parents=True, exist_ok=True)

  summary_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

  for page in range(num_pages):
    chunk = sample_df.iloc[page * per_page : (page + 1) * per_page]
    fig, axes = plt.subplots(4, 5, figsize=(18, 16.5))
    axes = axes.flatten()

    for i, (_, row) in enumerate(chunk.iterrows()):
      img_path = PROJECT_ROOT / row["img_filepath"]
      seg_path = PROJECT_ROOT / row["seg_filepath"]
      actual_label = int(row["label"])

      img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
      img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
      input_tensor = eval_tf(image=img_rgb)["image"].unsqueeze(0).to(device)

      # Inference (Pure Segmentation)
      seg_logits = model(input_tensor)
      image_prob = float(model.logits_to_image_prob(seg_logits).item())
      pred_label = 1 if image_prob >= threshold else 0

      # Mask ที่โมเดลวาดทำนาย (แสดงพิกเซลที่มั่นใจ > 0.5 หรือ > threshold)
      prob_map = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
      pred_mask = (prob_map > 0.5).astype(np.uint8)

      # Ground Truth Mask
      gt_mask = (cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

      # คำนวณ Dice เฉพาะเคสที่ป่วยจริง
      dice_str = ""
      if actual_label == 1:
        inter = np.sum(pred_mask * gt_mask)
        dice = (2.0 * inter + 1e-6) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-6)
        dice_str = f" | Dice:{dice:.2f}"

      if actual_label == 1 and pred_label == 1:
        tag, color = "TP", "#27AE60"
      elif actual_label == 0 and pred_label == 0:
        tag, color = "TN", "#27AE60"
      elif actual_label == 0 and pred_label == 1:
        tag, color = "FP", "#E67E22"
      else:
        tag, color = "FN", "#C0392B"
      summary_counts[tag] += 1

      ax = axes[i]
      ax.imshow(img_gray, cmap="gray")

      if pred_mask.sum() > 0:
        ax.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="cool", alpha=0.45)

      if gt_mask.sum() > 0:
          ax.contour(gt_mask, levels=[0.5], colors="#FF2E2E", linewidths=1.8)

      act_txt = "Pneumo" if actual_label == 1 else "Normal"
      pred_txt = "Urgent" if pred_label == 1 else "Normal"
      ax.set_title(
        f"[{tag}] Act:{act_txt} | Pred:{pred_txt}\n"
        f"p={image_prob:.3f}{dice_str}",
        fontsize=9,
        fontweight="bold",
        color=color,
        pad=5,
      )
      for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.2)
      ax.set_xticks([])
      ax.set_yticks([])

    for j in range(len(chunk), len(axes)):
      axes[j].axis("off")

    fig.suptitle(
      f"Test Set 20% Inspection (Page {page+1}/{num_pages}) — {args.model_name.upper()} (Thresh={threshold:.3f})\n"
      f"Red Outline = Ground Truth Mask | Cyan Fill = Model Predicted Mask (>0.5)",
      fontsize=13,
      fontweight="bold",
      y=1.01,
    )
    plt.tight_layout()
    save_path = out_dir / f"page_{page+1:02d}.png"
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

  print(f"[DONE] Saved {num_pages} pages to -> {out_dir}")
  print(f"[20% SAMPLE SUMMARY] {summary_counts}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize 20% of test set predictions vs ground truth masks")
  parser.add_argument("--model_name", type=str, default="densenet121")
  parser.add_argument("--frac", type=float, default=0.20, help="Fraction of test set to visualize")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  main(args)