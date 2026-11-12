import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import get_transforms
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GradCAM:
  """คำนวณ Grad-CAM Heatmap จาก Target Convolution Layer"""
  def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
    self.model = model
    self.gradients = None
    self.activations = None

    # ผูก Hook เพื่อดักจับ Feature Map และ Gradient
    target_layer.register_forward_hook(self._save_activation)
    target_layer.register_full_backward_hook(self._save_gradient)

  def _save_activation(self, module, input, output):
    self.activations = output.detach()

  def _save_gradient(self, module, grad_input, grad_output):
    self.gradients = grad_output[0].detach()

  def generate(self, input_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    self.model.zero_grad()
    logit = self.model(input_tensor)
    prob = float(torch.sigmoid(logit).item())

    # Backward หา Gradient ของ Logit เทียบกับ Feature Map สุดท้าย
    logit.backward()

    # Global Average Pooling บน Gradient (ตามสมการ Grad-CAM)
    weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
    cam = F.relu(cam)  # เก็บเฉพาะส่วนที่ส่งผลบวกต่อ Class 1

    # Upsample กลับไปที่ขนาดภาพต้นฉบับ (256x256)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize ให้อยู่ในช่วง [0, 1]
    if cam.max() > cam.min():
      cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
      cam = np.zeros_like(cam)

    return cam, prob


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_calibrated_threshold(model_name: str, default: float = 0.03) -> float:
  metrics_path = PROJECT_ROOT / "outputs" / "evaluations" / f"{model_name}_metrics.json"
  if metrics_path.exists():
    with open(metrics_path, "r") as f:
      return float(json.load(f)["clinical_triage"]["threshold"])
  return default


def main(args):
  device = get_device()
  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{args.model_name}_best.pth"
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
  img_size = checkpoint.get("img_size", 256)
  threshold = load_calibrated_threshold(args.model_name)

  model = build_model(model_name=args.model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  # เตรียม Grad-CAM จาก Target Layer ของโมเดล
  target_layer = model.get_gradcam_target_layer()
  gradcam = GradCAM(model, target_layer)

  # สุ่มเคส Pneumothorax (label=1) จาก test.csv จำนวน 5 ภาพ เพื่อเทียบ Mask vs Grad-CAM
  test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")
  pos_samples = test_df[test_df["label"] == 1].sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)

  _, eval_tf = get_transforms(img_size=img_size)

  fig, axes = plt.subplots(2, args.num_samples, figsize=(3.8 * args.num_samples, 7.8))

  for idx, row in pos_samples.iterrows():
    img_path = PROJECT_ROOT / row["img_filepath"]
    seg_path = PROJECT_ROOT / row["seg_filepath"]

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    input_tensor = eval_tf(image=img_rgb)["image"].unsqueeze(0).to(device)

    # สร้าง Grad-CAM Heatmap
    cam_map, prob = gradcam.generate(input_tensor)
    pred_label = 1 if prob >= threshold else 0
    status_text = "TP (Detected)" if pred_label == 1 else "FN (Missed)"
    status_color = "#27AE60" if pred_label == 1 else "#C0392B"

    # --- แถวที่ 1: Ground Truth Mask (เฉลยตำแหน่งลมรั่วจริง) ---
    ax_top = axes[0, idx]
    ax_top.imshow(img_gray, cmap="gray")
    if seg_path.exists():
      mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
      mask_bin = (mask > 0).astype(np.uint8)
      masked_overlay = np.ma.masked_where(mask_bin == 0, mask_bin)
      ax_top.imshow(masked_overlay, cmap="autumn", alpha=0.45)

    ax_top.set_title(f"Case #{idx+1} | Ground Truth\n(Red = Actual Pneumothorax)", fontsize=10, fontweight="bold")
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    # --- แถวที่ 2: Predicted Grad-CAM Heatmap (จุดที่โมเดลโฟกัส) ---
    ax_bot = axes[1, idx]
    ax_bot.imshow(img_gray, cmap="gray")
    ax_bot.imshow(cam_map, cmap="jet", alpha=0.45)

    ax_bot.set_title(
      f"Grad-CAM Attention\nProb: {prob:.3f} -> {status_text}",
      fontsize=10,
      fontweight="bold",
      color=status_color,
    )
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])

  fig.suptitle(
    f"Localization Check: Ground Truth Mask vs. Grad-CAM — {args.model_name.upper()} (Thresh={threshold:.4f})",
    fontsize=13,
    fontweight="bold",
    y=1.02,
  )
  plt.tight_layout()

  save_dir = PROJECT_ROOT / "outputs" / "gradcam"
  save_dir.mkdir(parents=True, exist_ok=True)
  save_path = save_dir / f"{args.model_name}_gt_vs_gradcam.png"
  plt.savefig(save_path, dpi=150, bbox_inches="tight")
  print(f"[SAVED] Grad-CAM comparison -> {save_path}")
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compare Ground Truth Mask vs Grad-CAM Attention")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture name")
  parser.add_argument("--num_samples", type=int, default=5, help="Number of positive test cases to inspect")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")

  args = parser.parse_args()
  main(args)