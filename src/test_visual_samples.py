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


def load_calibrated_threshold(model_name: str, default: float = 0.03) -> float:
  """ดึง Clinical Triage Threshold ที่คำนวณไว้จากไฟล์ evaluation JSON"""
  metrics_path = PROJECT_ROOT / "outputs" / "evaluations" / f"{model_name}_metrics.json"
  if metrics_path.exists():
    with open(metrics_path, "r") as f:
      data = json.load(f)
    return float(data["clinical_triage"]["threshold"])
  return default


@torch.no_grad()
def main(args):
  device = get_device()
  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.model_name}_best.pth"
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  # 1. โหลดโมเดลและ Clinical Threshold
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
  img_size = checkpoint.get("img_size", 256)
  threshold = load_calibrated_threshold(args.model_name)

  model = build_model(model_name=args.model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  print(f"[INFO] Loaded {args.model_name} | Clinical Triage Threshold = {threshold:.4f}")

  # 2. สุ่มภาพจาก test.csv (Class 0 จำนวน 5 ภาพ + Class 1 จำนวน 5 ภาพ)
  test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")
  n_half = args.num_samples // 2
  sample_neg = test_df[test_df["label"] == 0].sample(n=n_half, random_state=args.seed)
  sample_pos = test_df[test_df["label"] == 1].sample(n=args.num_samples - n_half, random_state=args.seed)
  samples_df = pd.concat([sample_neg, sample_pos], ignore_index=True)

  _, eval_tf = get_transforms(img_size=img_size)

  # 3. เตรียม Grid สำหรับ Plot (2 แถว x 5 คอลัมน์)
  cols = 5
  rows = int(np.ceil(args.num_samples / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(18, 4.3 * rows))
  axes = axes.flatten()

  for idx, (_, row) in enumerate(samples_df.iterrows()):
    img_path = PROJECT_ROOT / row["img_filepath"]
    seg_path = PROJECT_ROOT / row["seg_filepath"]
    actual_label = int(row["label"])

    # โหลดภาพสำหรับ Preprocess เข้าโมเดล (ให้ตรงกับ Pipeline ตอนเทรน/ประเมิน)
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    transformed = eval_tf(image=img_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # รัน Inference
    logit = model(input_tensor)
    prob = float(torch.sigmoid(logit).item())
    pred_label = 1 if prob >= threshold else 0

    # จัดกลุ่มผลลัพธ์ (TP, TN, FP, FN)
    if actual_label == 1 and pred_label == 1:
      outcome, border_color = "TP (Correct Urgent)", "#27AE60"
    elif actual_label == 0 and pred_label == 0:
      outcome, border_color = "TN (Correct Normal)", "#27AE60"
    elif actual_label == 0 and pred_label == 1:
      outcome, border_color = "FP (False Alarm)", "#E67E22"
    else:
      outcome, border_color = "FN (MISSED CASE!)", "#C0392B"

    # แสดงภาพ CXR พร้อมซ้อน Ground Truth Mask สีแดงโปร่งแสง (ถ้าป่วยจริง)
    ax = axes[idx]
    ax.imshow(img_gray, cmap="gray")
    if actual_label == 1 and seg_path.exists():
      mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
      mask_bin = (mask > 0).astype(np.uint8)
      masked_overlay = np.ma.masked_where(mask_bin == 0, mask_bin)
      ax.imshow(masked_overlay, cmap="autumn", alpha=0.4)

    actual_str = "Pneumo (1)" if actual_label == 1 else "Normal (0)"
    pred_str = "Urgent (1)" if pred_label == 1 else "Normal (0)"

    ax.set_title(
      f"Actual: {actual_str}\n"
      f"Pred: {pred_str} (p={prob:.3f})\n"
      f"[{outcome}]",
      fontsize=10,
      fontweight="bold",
      color=border_color,
      pad=8,
    )

    # ตีกรอบสีตามผลทำนายเพื่อให้อ่านง่าย
    for spine in ax.spines.values():
      spine.set_edgecolor(border_color)
      spine.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])

  # ซ่อนช่องที่เกิน (กรณี num_samples ไม่ลงตัวกับ 5)
  for j in range(len(samples_df), len(axes)):
    axes[j].axis("off")

  fig.suptitle(
    f"Test Set Visual Inspection — {args.model_name.upper()} (Triage Threshold = {threshold:.4f})\n"
    f"Top Row: Actual Normal (0) | Bottom Row: Actual Pneumothorax (1, Red Overlay = Ground Truth Mask)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
  )
  plt.tight_layout()

  save_dir = PROJECT_ROOT / "outputs" / "evaluations"
  save_dir.mkdir(parents=True, exist_ok=True)
  save_path = save_dir / f"{args.model_name}_test_10_samples.png"
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"[SAVED] Visual prediction grid -> {save_path}")
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize 10 random test predictions vs ground truth")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture name")
  parser.add_argument("--num_samples", type=int, default=10, help="Total samples to plot (half neg, half pos)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

  args = parser.parse_args()
  main(args)