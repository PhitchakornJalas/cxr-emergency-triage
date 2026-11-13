import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import random
from pathlib import Path

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXRPneumothoraxDataset, get_transforms
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int = 42):
  """ล็อก Random Seed ทุกส่วนเพื่อให้ผลการทดลองทำซ้ำได้เหมือนเดิม"""
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  if torch.backends.mps.is_available():
    torch.backends.mps.deterministic = False
    torch.mps.manual_seed(seed)


def get_device() -> torch.device:
  """เลือก Device อัตโนมัติ (CUDA -> Apple MPS -> CPU)"""
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, stage: int, accum_steps: int = 1):
  model.train()
  running_loss = 0.0
  optimizer.zero_grad()

  for step, (X, y) in enumerate(tqdm(loader, desc=f"Train S{stage}", leave=False)):
    images = X.to(device)
    labels = y.float().unsqueeze(1).to(device)  # shape: [B, 1]

    # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
    if stage == 1:
      logits = model(images, apply_bsn=False)
      loss = criterion(logits, labels)
    else:
      # Stage 2: เปิด Batch Score Normalization (BSN) ก่อนเข้า AUCMLoss ตามเปเปอร์
      preds = model(images, apply_bsn=True)
      loss = criterion(preds, labels)

    loss_scaled = loss / accum_steps

    loss_scaled.backward()

    if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
      optimizer.step()
      optimizer.zero_grad()

    running_loss += loss.item() * images.size(0)

  if device.type == "mps":
    torch.mps.empty_cache()

  return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, stage: int):
  model.eval()
  running_loss = 0.0
  all_probs = []
  all_targets = []

  for X, y in tqdm(loader, desc=f"Val   S{stage}", leave=False):
    images = X.to(device)
    labels = y.float().unsqueeze(1).to(device)

    # คำนวณ Validation Loss ให้ตรงกับ Stage
    if stage == 1:
      logits = model(images, apply_bsn=False)
      loss = criterion(logits, labels)
    else:
      logits = model(images, apply_bsn=False)
      bsn_preds = model(images, apply_bsn=True)
      loss = criterion(bsn_preds, labels)

    # ตอนวัดผล ROC-AUC / PR-AUC ใช้ความน่าจะเป็นดิบ (ปิด BSN เพื่อไม่ให้ขึ้นกับขนาด Batch)
    probs = torch.sigmoid(logits).squeeze(1)

    running_loss += loss.item() * images.size(0)
    all_probs.extend(probs.cpu().numpy())
    all_targets.extend(y.numpy())

  val_loss = running_loss / len(loader.dataset)
  val_roc_auc = roc_auc_score(all_targets, all_probs)
  val_pr_auc = average_precision_score(all_targets, all_probs)

  if device.type == "mps":
    torch.mps.empty_cache()

  return val_loss, val_roc_auc, val_pr_auc


def main(args):
  seed_everything(args.seed)
  device = get_device()
  print(f"[INFO] DAM Classification (Stage {args.stage}) | Device: {device} | Model: {args.model_name} | Seed: {args.seed}")

  train_csv = PROJECT_ROOT / "data" / "train.csv"
  val_csv = PROJECT_ROOT / "data" / "val.csv"

  train_tf, val_tf = get_transforms(img_size=args.img_size)
  train_dataset = CXRPneumothoraxDataset(train_csv, transform=train_tf)
  val_dataset = CXRPneumothoraxDataset(val_csv, transform=val_tf)

  g = torch.Generator()
  g.manual_seed(args.seed)

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
    prefetch_factor=2 if args.num_workers > 0 else None,
    generator=g,
    drop_last=(args.stage == 2),  # กัน Batch สุดท้ายเหลือ 1 รูปซึ่งจะทำให้ L2 Norm เพี้ยน
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0),
  )

  ckpt_dir = PROJECT_ROOT / "checkpoints"
  log_dir = PROJECT_ROOT / "outputs" / "logs"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)

  model = build_model(model_name=args.model_name, pretrained=True, drop_rate=args.drop_rate).to(device)

  if args.stage == 1:
    # --- STAGE 1: Regular Pretraining ด้วย Cross-Entropy (หัวข้อ 4.2.1) ---
    criterion = nn.BCEWithLogitsLoss()
    lr = args.lr if args.lr is not None else 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    accum_steps = 2
    best_ckpt_path = ckpt_dir / f"{args.model_name}_dam_stage1_best.pth"
    history_path = log_dir / f"{args.model_name}_dam_stage1_history.csv"

  elif args.stage == 2:
    # --- STAGE 2: Deep AUC Maximization ด้วย AUC Margin Loss + PESG ---
    stage1_ckpt_path = ckpt_dir / f"{args.model_name}_dam_stage1_best.pth"
    if not stage1_ckpt_path.exists():
      raise FileNotFoundError(f"ไม่พบไฟล์ Stage 1 checkpoint ที่ {stage1_ckpt_path} กรุณารัน --stage 1 ก่อน")

    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.reset_classifier()  # ตัดหัว Classifier เดิมทิ้ง สร้างใหม่แบบ Random Init
    model = model.to(device)

    imratio = train_dataset.get_imratio()
    print(f"[INFO] Stage 2 Loaded Stage 1 Weights (Val AUC: {ckpt['val_roc_auc']:.4f}) | Train Imratio: {imratio:.4f}")

    lr = args.lr if args.lr is not None else 0.05
    criterion = AUCMLoss(imratio=imratio, margin=args.margin).to(device)
    optimizer = PESG(
      model.parameters(),
      loss_fn=criterion,
      lr=lr,
      gamma=args.gamma,
      margin=args.margin,
      weight_decay=0.0,
    )
    scheduler = None
    accum_steps = 2  # ห้ามใช้ gradient accumulation ใน Stage 2 เพราะ BSN ผูกกับขนาด Batch
    best_ckpt_path = ckpt_dir / f"{args.model_name}_dam_stage2_best.pth"
    history_path = log_dir / f"{args.model_name}_dam_stage2_history.csv"

  best_roc_auc = 0.0
  no_improve_epochs = 0
  history = []
  decay_epochs = {int(args.epochs * 0.5), int(args.epochs * 0.75)}  # ลด LR ลง 3 เท่าที่ 50% และ 75% ของ Epochs

  print("-" * 80)
  for epoch in range(1, args.epochs + 1):
    if args.stage == 2 and epoch in decay_epochs:
      optimizer.update_regularizer(decay_factor=3)
      print(f"[INFO] PESG Learning Rate decayed by 3x at Epoch {epoch}")

    current_lr = optimizer.param_groups[0]["lr"]
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, stage=args.stage, accum_steps=accum_steps)
    val_loss, val_roc_auc, val_pr_auc = validate(model, val_loader, criterion, device, stage=args.stage)

    if scheduler is not None:
      scheduler.step()

    history.append({
      "epoch": epoch,
      "lr": current_lr,
      "train_loss": train_loss,
      "val_loss": val_loss,
      "val_roc_auc": val_roc_auc,
      "val_pr_auc": val_pr_auc,
    })

    saved_flag = ""
    if val_roc_auc > best_roc_auc:
      best_roc_auc = val_roc_auc
      no_improve_epochs = 0
      torch.save({
        "stage": args.stage,
        "epoch": epoch,
        "model_name": args.model_name,
        "img_size": args.img_size,
        "model_state_dict": model.state_dict(),
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
      }, best_ckpt_path)
      saved_flag = " <-- [BEST SAVED]"
    else:
      no_improve_epochs += 1

    print(
      f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
      f"LR: {current_lr:.2e} | "
      f"Train Loss: {train_loss:.4f} | "
      f"Val Loss: {val_loss:.4f} | "
      f"ROC-AUC: {val_roc_auc:.4f} | "
      f"PR-AUC: {val_pr_auc:.4f}{saved_flag}"
    )

    if no_improve_epochs >= args.patience:
      print(f"\n[EARLY STOPPING] Val ROC-AUC did not improve for {args.patience} epochs. Stopping early.")
      break

  history_df = pd.DataFrame(history)
  history_df.to_csv(history_path, index=False)

  print("-" * 80)
  print(f"[FINISH Stage {args.stage}] Best Val ROC-AUC: {best_roc_auc:.4f}")
  print(f"[SAVED] Checkpoint : {best_ckpt_path}")
  print(f"[SAVED] Train Log  : {history_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train Deep AUC Maximization (DAM) CXR Triage Model")
  parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1: CE Pretraining, 2: AUC Margin Loss")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture from timm")
  parser.add_argument("--img_size", type=int, default=256, help="Input image size")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size (32 recommended for BSN stability)")
  parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
  parser.add_argument("--lr", type=float, default=None, help="Initial LR (default: 1e-5 for Stage 1, 0.05 for Stage 2)")
  parser.add_argument("--margin", type=float, default=0.8, help="Margin m for AUCMLoss (0.8 in CheXpert paper)")
  parser.add_argument("--gamma", type=float, default=500, help="Gamma regularizer for PESG")
  parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout rate")
  parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (0 is safest on macOS)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs")

  args = parser.parse_args()
  main(args)