import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
  average_precision_score,
  confusion_matrix,
  f1_score,
  precision_recall_curve,
  roc_auc_score,
  roc_curve,
)
import torch
from torch.utils.data import DataLoader
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


@torch.no_grad()
def predict_outputs(model: torch.nn.Module, loader: DataLoader, device: torch.device):
  """รัน Inference คืนค่า Image-level Labels และ Probs (ปิด BSN เพื่อให้ค่าคงที่ทุกรูป)"""
  model.eval()
  all_probs = []
  all_targets = []

  for X, labels in tqdm(loader, desc="Inference", leave=False):
    images = X.to(device)
    logits = model(images, apply_bsn=False)
    probs = torch.sigmoid(logits).squeeze(1)

    all_probs.extend(probs.cpu().numpy())
    all_targets.extend(labels.numpy())

  return np.array(all_targets), np.array(all_probs)


def find_clinical_threshold(y_true: np.ndarray, y_probs: np.ndarray, target_sensitivity: float = 0.95) -> float:
  """
  หา Threshold สูงสุดบน Validation Set ที่ยังคงให้ค่า Sensitivity >= target_sensitivity
  (เพื่อลด False Positive ให้มากที่สุด ภายใต้เงื่อนไขว่าห้ามหลุดเคสฉุกเฉินเกิน 5%)
  """
  thresholds = np.linspace(0.01, 0.99, 981)
  best_thresh = 0.01

  for t in thresholds:
    preds = (y_probs >= t).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fn = np.sum((preds == 0) & (y_true == 1))
    sensitivity = tp / (tp + fn + 1e-8)

    if sensitivity >= target_sensitivity:
      best_thresh = float(t)
    else:
      break

  return best_thresh


def compute_clinical_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
  """คำนวณตัวชี้วัดทางการแพทย์ครบชุด ณ Threshold ที่กำหนด"""
  preds = (y_probs >= threshold).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

  sensitivity = tp / (tp + fn + 1e-8)
  specificity = tn / (tn + fp + 1e-8)
  ppv = tp / (tp + fp + 1e-8)
  npv = tn / (tn + fn + 1e-8)
  f1 = f1_score(y_true, preds)
  roc_auc = roc_auc_score(y_true, y_probs)
  pr_auc = average_precision_score(y_true, y_probs)

  return {
    "threshold": round(threshold, 4),
    "roc_auc": round(float(roc_auc), 4),
    "pr_auc": round(float(pr_auc), 4),
    "sensitivity": round(float(sensitivity), 4),
    "specificity": round(float(specificity), 4),
    "ppv_precision": round(float(ppv), 4),
    "npv": round(float(npv), 4),
    "f1_score": round(float(f1), 4),
    "tp": int(tp),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
  }


def plot_evaluation_curves(
  y_true: np.ndarray,
  y_probs: np.ndarray,
  metrics: dict,
  model_name: str,
  stage: int,
  save_path: Path,
):
  """พล็อตกราฟ 3 ช่อง: ROC Curve | Precision-Recall Curve | Clinical Confusion Matrix"""
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

  # 1. ROC Curve
  fpr, tpr, _ = roc_curve(y_true, y_probs)
  axes[0].plot(fpr, tpr, color="#2980B9", lw=2, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
  axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
  axes[0].scatter(
    1 - metrics["specificity"],
    metrics["sensitivity"],
    color="#E74C3C",
    zorder=5,
    label=f"Triage Point (Sens={metrics['sensitivity']:.2f})",
  )
  axes[0].set_title("ROC Curve (Test Set)", fontweight="bold")
  axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
  axes[0].set_ylabel("True Positive Rate (Sensitivity)")
  axes[0].legend(loc="lower right")
  axes[0].grid(alpha=0.3)

  # 2. Precision-Recall Curve
  precision, recall, _ = precision_recall_curve(y_true, y_probs)
  axes[1].plot(recall, precision, color="#27AE60", lw=2, label=f"PR-AUC = {metrics['pr_auc']:.4f}")
  axes[1].scatter(
    metrics["sensitivity"],
    metrics["ppv_precision"],
    color="#E74C3C",
    zorder=5,
    label=f"Triage Point (PPV={metrics['ppv_precision']:.2f})",
  )
  axes[1].set_title("Precision-Recall Curve (Test Set)", fontweight="bold")
  axes[1].set_xlabel("Recall (Sensitivity)")
  axes[1].set_ylabel("Precision (PPV)")
  axes[1].legend(loc="lower left")
  axes[1].grid(alpha=0.3)

  # 3. Confusion Matrix
  cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
  sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    ax=axes[2],
    xticklabels=["Pred: Normal (0)", "Pred: Urgent (1)"],
    yticklabels=["Actual: Normal (0)", "Actual: Pneumo (1)"],
    annot_kws={"size": 12, "weight": "bold"},
  )
  axes[2].set_title(f"Confusion Matrix (Thresh = {metrics['threshold']})", fontweight="bold")

  fig.suptitle(
    f"Clinical Triage Evaluation — {model_name} (DAM Stage {stage})",
    fontsize=13,
    fontweight="bold",
    y=1.02,
  )
  plt.tight_layout()
  save_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"[SAVED] Evaluation plot -> {save_path}")
  plt.close()


def main(args):
  device = get_device()
  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.model_name}_dam_stage{args.stage}_best.pth"
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  img_size = checkpoint.get("img_size", 256)

  model = build_model(model_name=args.model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  print(
    f"[INFO] Loaded {args.model_name} (DAM Stage {args.stage}) | Best Epoch {checkpoint['epoch']} "
    f"| Val ROC-AUC: {checkpoint['val_roc_auc']:.4f} | Val PR-AUC: {checkpoint.get('val_pr_auc', 0.0):.4f}"
  )

  _, eval_tf = get_transforms(img_size=img_size)
  val_dataset = CXRPneumothoraxDataset(PROJECT_ROOT / "data" / "val.csv", transform=eval_tf)
  test_dataset = CXRPneumothoraxDataset(PROJECT_ROOT / "data" / "test.csv", transform=eval_tf)

  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

  val_targets, val_probs = predict_outputs(model, val_loader, device)
  calibrated_thresh = find_clinical_threshold(val_targets, val_probs, target_sensitivity=args.target_sens)

  test_targets, test_probs = predict_outputs(model, test_loader, device)
  default_metrics = compute_clinical_metrics(test_targets, test_probs, threshold=0.50)
  triage_metrics = compute_clinical_metrics(test_targets, test_probs, threshold=calibrated_thresh)

  comparison_df = pd.DataFrame([
    {"Mode": "Default (0.50)", **default_metrics},
    {"Mode": f"Clinical Triage (Sens>={int(args.target_sens * 100)}%)", **triage_metrics},
  ])

  print("\n" + "=" * 95)
  print(f"TEST SET EVALUATION SUMMARY: {args.model_name.upper()} (DAM STAGE {args.stage})")
  print("=" * 95)
  print(comparison_df.to_string(index=False))
  print("=" * 95)

  eval_dir = PROJECT_ROOT / "outputs" / "evaluations"
  eval_dir.mkdir(parents=True, exist_ok=True)

  result_payload = {
    "model_name": args.model_name,
    "architecture": f"dam_stage{args.stage}_classification",
    "best_epoch": int(checkpoint["epoch"]),
    "val_roc_auc": round(float(checkpoint["val_roc_auc"]), 4),
    "default_050": default_metrics,
    "clinical_triage": triage_metrics,
  }
  json_path = eval_dir / f"{args.model_name}_dam_stage{args.stage}_metrics.json"
  with open(json_path, "w") as f:
    json.dump(result_payload, f, indent=2)
  print(f"[SAVED] Metrics JSON    -> {json_path}")

  plot_path = eval_dir / f"{args.model_name}_dam_stage{args.stage}_eval_curves.png"
  plot_evaluation_curves(test_targets, test_probs, triage_metrics, args.model_name, args.stage, plot_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Evaluate & Calibrate DAM CXR Triage Model")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture name")
  parser.add_argument("--stage", type=int, choices=[1, 2], default=2, help="Evaluate Stage 1 or Stage 2 checkpoint")
  parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
  parser.add_argument("--target_sens", type=float, default=0.95, help="Target sensitivity for triage calibration")

  args = parser.parse_args()
  main(args)