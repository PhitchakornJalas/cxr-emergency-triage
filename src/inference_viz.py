import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import CXRPneumothoraxDataset, get_transforms
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
  """แปลง Normalized Tensor [3, H, W] กลับเป็นภาพ RGB uint8 [H, W, 3]"""
  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  img = tensor.cpu().numpy().transpose(1, 2, 0)
  img = (img * std + mean) * 255.0
  return np.clip(img, 0, 255).astype(np.uint8)


def create_overlay(image_rgb: np.ndarray, mask_2d: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
  """ระบายสีโปร่งแสงทับลงบนภาพเอกซเรย์เฉพาะบริเวณที่ Mask > 0.5"""
  overlay = image_rgb.copy()
  binary_mask = mask_2d > 0.5
  if np.any(binary_mask):
    color_layer = np.zeros_like(image_rgb, dtype=np.uint8)
    color_layer[:] = color_rgb
    overlay[binary_mask] = cv2.addWeighted(
      image_rgb[binary_mask], 1.0 - alpha, color_layer[binary_mask], alpha, 0
    )
    # วาดเส้นขอบให้เห็นชัดเจนขึ้น
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_rgb, thickness=2)
  return overlay


def load_calibrated_threshold(model_name: str, stage: int, default_thresh: float = 0.009) -> float:
  """ดึงค่า Clinical Triage Threshold ที่คำนวณไว้จาก evaluate.py อัตโนมัติ"""
  json_path = PROJECT_ROOT / "outputs" / "evaluations" / f"{model_name}_hybrid_stage{stage}_metrics.json"
  if json_path.exists():
    with open(json_path, "r") as f:
      data = json.load(f)
    thresh = data.get("clinical_triage", {}).get("threshold", default_thresh)
    print(f"[INFO] Loaded calibrated triage threshold from JSON: {thresh:.4f}")
    return float(thresh)
  print(f"[WARN] Metrics JSON not found, using fallback threshold: {default_thresh:.4f}")
  return default_thresh


@torch.no_grad()
def run_visualization(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  device = get_device()

  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.model_name}_hybrid_stage{args.stage}_best.pth"
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  img_size = checkpoint.get("img_size", 256)

  model = build_model(model_name=args.model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  threshold = args.threshold if args.threshold is not None else load_calibrated_threshold(args.model_name, args.stage)

  _, eval_tf = get_transforms(img_size=img_size)
  test_dataset = CXRPneumothoraxDataset(PROJECT_ROOT / "data" / "test.csv", transform=eval_tf, load_mask=True)

  total_test = len(test_dataset)
  sample_count = max(1, int(round(total_test * args.sample_ratio)))

  # สุ่มตัวอย่างแบบ Stratified (รักษาสัดส่วนเคสป่วย vs ปกติ 10% ของ Test Set)
  pos_indices = np.where(test_dataset.targets == 1)[0].tolist()
  neg_indices = np.where(test_dataset.targets == 0)[0].tolist()

  n_pos = max(1, int(round(len(pos_indices) * args.sample_ratio)))
  n_neg = max(1, sample_count - n_pos)

  sampled_indices = random.sample(pos_indices, min(n_pos, len(pos_indices))) + random.sample(neg_indices, min(n_neg, len(neg_indices)))
  random.shuffle(sampled_indices)

  print(f"[INFO] Sampling {len(sampled_indices)} / {total_test} test images ({args.sample_ratio * 100:.0f}%): {n_pos} Positive, {n_neg} Negative")

  out_base_dir = PROJECT_ROOT / "outputs" / "visualizations" / f"{args.model_name}_stage{args.stage}"
  for sub in ["TP_Urgent_Correct", "FN_Missed_Urgent", "FP_False_Alarm", "TN_Normal_Correct"]:
    (out_base_dir / sub).mkdir(parents=True, exist_ok=True)

  summary_records = []

  for idx in tqdm(sampled_indices, desc="Generating Viz"):
    X, y = test_dataset[idx]
    gt_label = int(y["label"].item())
    gt_mask = y["mask"].squeeze(0).numpy()
    orig_filepath = test_dataset.df.iloc[idx]["img_filepath"]
    img_stem = Path(orig_filepath).stem

    input_tensor = X.unsqueeze(0).to(device)
    cls_logits, seg_logits = model(input_tensor, apply_bsn=False, run_decoder=True)
    triage_score = float(model.fuse_triage_probs(cls_logits, seg_logits, top_k=16, alpha=0.7).item())
    pred_mask_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

    pred_label = 1 if triage_score >= threshold else 0

    # จัดหมวดหมู่ผลลัพธ์ (TP / FN / FP / TN)
    if gt_label == 1 and pred_label == 1:
      category = "TP_Urgent_Correct"
      status_text = "URGENT (Correct Detection)"
      badge_color = "#27AE60"
    elif gt_label == 1 and pred_label == 0:
      category = "FN_Missed_Urgent"
      status_text = "MISSED (False Negative)"
      badge_color = "#C0392B"
    elif gt_label == 0 and pred_label == 1:
      category = "FP_False_Alarm"
      status_text = "URGENT (False Positive)"
      badge_color = "#E67E22"
    else:
      category = "TN_Normal_Correct"
      status_text = "NORMAL (Correct)"
      badge_color = "#2980B9"

    # คำนวณ Dice เฉพาะเคสที่มีรอยโรคจริง
    if gt_label == 1:
      pred_bin = (pred_mask_prob > 0.5).astype(np.float32)
      inter = (pred_bin * gt_mask).sum()
      dice_score = (2.0 * inter + 1e-6) / (pred_bin.sum() + gt_mask.sum() + 1e-6)
      dice_str = f" | Mask Dice: {dice_score:.3f}"
    else:
      dice_score = None
      dice_str = ""

    img_rgb = denormalize_image(X)
    gt_overlay = create_overlay(img_rgb, gt_mask, color_rgb=(46, 204, 113), alpha=0.45)       # สีเขียว = เฉลยแพทย์
    pred_overlay = create_overlay(img_rgb, pred_mask_prob, color_rgb=(231, 76, 60), alpha=0.45)  # สีแดง = AI ทำนาย

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"1. Original CXR\n({img_stem})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    gt_title = "2. Ground Truth: PNEUMOTHORAX (1)" if gt_label == 1 else "2. Ground Truth: NORMAL (0)"
    axes[1].imshow(gt_overlay)
    axes[1].set_title(gt_title, fontsize=10, fontweight="bold", color="#1E8449" if gt_label == 1 else "#2C3E50")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(
      f"3. AI Triage: {status_text}\nScore: {triage_score:.4f} (Thresh: {threshold:.4f}){dice_str}",
      fontsize=10,
      fontweight="bold",
      color=badge_color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    save_file = out_base_dir / category / f"{img_stem}_score_{triage_score:.4f}.png"
    plt.savefig(save_file, dpi=130, bbox_inches="tight")
    plt.close(fig)

    summary_records.append({
      "image_id": img_stem,
      "category": category,
      "gt_label": gt_label,
      "pred_label": pred_label,
      "triage_score": round(triage_score, 4),
      "threshold": round(threshold, 4),
      "dice_score": round(float(dice_score), 4) if dice_score is not None else np.nan,
      "saved_path": str(save_file.relative_to(PROJECT_ROOT)),
    })

  summary_df = pd.DataFrame(summary_records)
  csv_path = out_base_dir / "sampled_10pct_predictions.csv"
  summary_df.to_csv(csv_path, index=False)

  print("\n" + "=" * 65)
  print(f"[FINISH] Generated {len(summary_df)} visualization images (10% of Test Set)")
  print(summary_df["category"].value_counts().to_string())
  print(f"[SAVED] Images Folder : {out_base_dir}")
  print(f"[SAVED] Summary CSV   : {csv_path}")
  print("=" * 65)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize Hybrid Seg+DAM Predictions on 10% Sampled Test Set")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture name")
  parser.add_argument("--stage", type=int, choices=[1, 2], default=2, help="Model stage (1 or 2)")
  parser.add_argument("--sample_ratio", type=float, default=0.10, help="Fraction of test set to sample (default: 0.10 = 10%)")
  parser.add_argument("--threshold", type=float, default=None, help="Manual threshold (default: auto-load from metrics.json)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")

  args = parser.parse_args()
  run_visualization(args)