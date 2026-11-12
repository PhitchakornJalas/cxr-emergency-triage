import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CXRPneumothoraxDataset, get_transforms
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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
    torch.backends.mps.deterministic = True
    torch.mps.manual_seed(seed)


def get_device() -> torch.device:
  """เลือก Device อัตโนมัติ (CUDA -> Apple MPS -> CPU)"""
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, accum_steps: int = 2):
  model.train()
  running_loss = 0.0
  optimizer.zero_grad()

  for step, (images, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
    images = images.to(device)
    labels = labels.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
      logits = model(images)
      loss = criterion(logits, labels)
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
def validate(model, loader, criterion, device):
  model.eval()
  running_loss = 0.0
  all_probs = []
  all_targets = []

  for images, labels in tqdm(loader, desc="Val  ", leave=False):
    images = images.to(device)
    labels = labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)
    probs = torch.sigmoid(logits)

    running_loss += loss.item() * images.size(0)
    all_probs.extend(probs.cpu().numpy())
    all_targets.extend(labels.cpu().numpy())

  val_loss = running_loss / len(loader.dataset)
  val_roc_auc = roc_auc_score(all_targets, all_probs)
  val_pr_auc = average_precision_score(all_targets, all_probs)

  if device.type == "mps":
    torch.mps.empty_cache()

  return val_loss, val_roc_auc, val_pr_auc


def main(args):
  seed_everything(args.seed)
  device = get_device()
  print(f"[INFO] Device: {device} | Model: {args.model_name} | Seed: {args.seed}")

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
    generator=g
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    persistent_workers=(args.num_workers > 0)
  )

  train_df = pd.read_csv(train_csv)
  num_pos = (train_df["label"] == 1).sum()
  num_neg = (train_df["label"] == 0).sum()
  pos_weight_val = num_neg / num_pos
  print(f"[INFO] Class Ratio (Neg/Pos): {num_neg}/{num_pos} -> pos_weight = {pos_weight_val:.2f}")

  pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

  model = build_model(model_name=args.model_name, pretrained=True, drop_rate=args.drop_rate).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

  ckpt_dir = PROJECT_ROOT / "checkpoints"
  log_dir = PROJECT_ROOT / "outputs" / "logs"
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)

  best_ckpt_path = ckpt_dir / f"{args.model_name}_best.pth"
  best_roc_auc = 0.0
  no_improve_epochs = 0
  history = []

  print("-" * 75)
  for epoch in range(1, args.epochs + 1):
    current_lr = optimizer.param_groups[0]["lr"]
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_roc_auc, val_pr_auc = validate(model, val_loader, criterion, device)
    scheduler.step()

    history.append({
      "epoch": epoch,
      "lr": current_lr,
      "train_loss": train_loss,
      "val_loss": val_loss,
      "val_roc_auc": val_roc_auc,
      "val_pr_auc": val_pr_auc
    })

    saved_flag = ""
    if val_roc_auc > best_roc_auc:
      best_roc_auc = val_roc_auc
      no_improve_epochs = 0
      torch.save({
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
  history_path = log_dir / f"{args.model_name}_history.csv"
  history_df.to_csv(history_path, index=False)

  print("-" * 75)
  print(f"[FINISH] Best Val ROC-AUC: {best_roc_auc:.4f}")
  print(f"[SAVED]  Checkpoint : {best_ckpt_path}")
  print(f"[SAVED]  Train Log  : {history_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train CXR Emergency Triage Model")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture from timm")
  parser.add_argument("--img_size", type=int, default=256, help="Input image size")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
  parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
  parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
  parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout rate")
  parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 is safest on macOS)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs")

  args = parser.parse_args()
  main(args)