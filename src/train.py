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
from libauc.sampler import DualSampler
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
    torch.mps.manual_seed(seed)


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


class MultiTaskStage1Loss(nn.Module):
  def __init__(self, smooth: float = 1.0):
    super().__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.smooth = smooth

  def forward(self, cls_logits: torch.Tensor, labels: torch.Tensor, seg_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    cls_loss = self.bce(cls_logits, labels)
    seg_bce = self.bce(seg_logits, masks)

    probs = torch.sigmoid(seg_logits).view(seg_logits.size(0), -1)
    targets = masks.view(masks.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)).mean()

    seg_loss = 0.5 * seg_bce + 0.5 * dice_loss
    return 0.5 * seg_loss + 0.5 * cls_loss


def train_one_epoch(model, loader, criterion, optimizer, device, stage: int, accum_steps: int = 1):
  model.train()
  if stage == 2:
    # คง BatchNorm ใน Decoder ให้อยู่ในโหมด eval ระหว่างเทรน Stage 2
    for m in [model.dec3, model.dec2, model.dec1]:
      m.eval()

  running_loss = 0.0
  optimizer.zero_grad()

  for step, (X, y) in enumerate(tqdm(loader, desc=f"Train S{stage}", leave=False)):
    images = X.to(device)
    labels = y["label"].float().unsqueeze(1).to(device)

    if stage == 1:
      masks = y["mask"].to(device)
      cls_logits, seg_logits = model(images, apply_bsn=False, run_decoder=True)
      loss = criterion(cls_logits, labels, seg_logits, masks)
    else:
      # Stage 2: ปิดการคำนวณ Decoder ชั่วคราวเพื่อความเร็วสูงสุด (Weights ถูกล็อกไว้แล้ว)
      preds, _ = model(images, apply_bsn=True, run_decoder=False)
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
def validate(model, loader, device, stage: int):
  model.eval()
  running_loss = 0.0
  all_probs = []
  all_targets = []
  pos_dice_scores = []
  val_bce = nn.BCEWithLogitsLoss()

  for X, y in tqdm(loader, desc=f"Val   S{stage}", leave=False):
    images = X.to(device)
    labels_cpu = y["label"]
    labels = labels_cpu.float().unsqueeze(1).to(device)
    masks = y["mask"].to(device)

    cls_logits, seg_logits = model(images, apply_bsn=False, run_decoder=True)
    loss = val_bce(cls_logits, labels)

    # รวมคะแนนจาก DAM Classifier (0.7) + Segmentation Mask Top-k (0.3)
    probs = model.fuse_triage_probs(cls_logits, seg_logits, top_k=16, alpha=0.7)

    running_loss += loss.item() * images.size(0)
    all_probs.extend(probs.cpu().numpy())
    all_targets.extend(labels_cpu.numpy())

    pred_masks = (torch.sigmoid(seg_logits) > 0.5).float()
    for i in range(images.size(0)):
      if labels_cpu[i].item() == 1.0:
        inter = (pred_masks[i] * masks[i]).sum().item()
        denom = pred_masks[i].sum().item() + masks[i].sum().item()
        pos_dice_scores.append((2.0 * inter + 1e-6) / (denom + 1e-6))

  val_loss = running_loss / len(loader.dataset)
  val_roc_auc = roc_auc_score(all_targets, all_probs)
  val_pr_auc = average_precision_score(all_targets, all_probs)
  val_pos_dice = float(np.mean(pos_dice_scores)) if pos_dice_scores else 0.0

  if device.type == "mps":
    torch.mps.empty_cache()

  return val_loss, val_roc_auc, val_pr_auc, val_pos_dice


def main(args):
  seed_everything(args.seed)
  device = get_device()
  print(f"[INFO] Hybrid Seg+DAM (Stage {args.stage}) | Device: {device} | Model: {args.model_name}")

  train_csv = PROJECT_ROOT / "data" / "train.csv"
  val_csv = PROJECT_ROOT / "data" / "val.csv"

  train_tf, val_tf = get_transforms(img_size=args.img_size)
  train_dataset = CXRPneumothoraxDataset(train_csv, transform=train_tf, load_mask=(args.stage == 1))
  val_dataset = CXRPneumothoraxDataset(val_csv, transform=val_tf, load_mask=True)

  g = torch.Generator()
  g.manual_seed(args.seed)

  if args.stage == 2:
    sampler = DualSampler(train_dataset, batch_size=args.batch_size, sampling_rate=0.25)
    train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      sampler=sampler,
      num_workers=args.num_workers,
      persistent_workers=(args.num_workers > 0),
      drop_last=True,
    )
  else:
    train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      persistent_workers=(args.num_workers > 0),
      generator=g,
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
    criterion = MultiTaskStage1Loss()
    lr = args.lr if args.lr is not None else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    accum_steps = 2
    best_ckpt_path = ckpt_dir / f"{args.model_name}_hybrid_stage1_best.pth"
    history_path = log_dir / f"{args.model_name}_hybrid_stage1_history.csv"

  elif args.stage == 2:
    stage1_ckpt_path = ckpt_dir / f"{args.model_name}_hybrid_stage1_best.pth"
    if not stage1_ckpt_path.exists():
      raise FileNotFoundError(f"ไม่พบไฟล์ Stage 1 ที่ {stage1_ckpt_path} กรุณารัน --stage 1 ก่อน")

    ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.reset_classifier_and_freeze_decoder()
    model = model.to(device)

    imratio = train_dataset.get_imratio()
    print(f"[INFO] Loaded Stage 1 (Val AUC: {ckpt['val_roc_auc']:.4f} | Pos Dice: {ckpt['val_pos_dice']:.4f})")

    # ใช้ lr ต่ำลง (0.01) เพื่อจูน AUC Margin Loss โดยไม่กระทบ Feature Maps ของ Decoder
    lr = args.lr if args.lr is not None else 0.01
    criterion = AUCMLoss(imratio=imratio, margin=args.margin, device=device).to(device)
    criterion.a.data = criterion.a.data.to(device)
    criterion.b.data = criterion.b.data.to(device)
    criterion.alpha.data = criterion.alpha.data.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
      optimizer = PESG(
        trainable_params,
        loss_fn=criterion,
        lr=lr,
        gamma=args.gamma,
        margin=args.margin,
        weight_decay=0.0,
        device=device,
      )
    except TypeError:
      optimizer = PESG(
        trainable_params,
        loss_fn=criterion,
        lr=lr,
        gamma=args.gamma,
        margin=args.margin,
        weight_decay=0.0,
      )

    scheduler = None
    accum_steps = 1
    best_ckpt_path = ckpt_dir / f"{args.model_name}_hybrid_stage2_best.pth"
    history_path = log_dir / f"{args.model_name}_hybrid_stage2_history.csv"

  best_roc_auc = 0.0
  no_improve_epochs = 0
  history = []
  decay_epochs = {int(args.epochs * 0.5), int(args.epochs * 0.75)}

  print("-" * 90)
  for epoch in range(1, args.epochs + 1):
    if args.stage == 2 and epoch in decay_epochs:
      optimizer.update_regularizer(decay_factor=3)
      print(f"[INFO] PESG Learning Rate decayed by 3x at Epoch {epoch}")

    current_lr = optimizer.param_groups[0]["lr"]
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, stage=args.stage, accum_steps=accum_steps)
    val_loss, val_roc_auc, val_pr_auc, val_pos_dice = validate(model, val_loader, device, stage=args.stage)

    if scheduler is not None:
      scheduler.step()

    history.append({
      "epoch": epoch,
      "lr": current_lr,
      "train_loss": train_loss,
      "val_loss": val_loss,
      "val_roc_auc": val_roc_auc,
      "val_pr_auc": val_pr_auc,
      "val_pos_dice": val_pos_dice,
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
        "val_pos_dice": val_pos_dice,
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
      f"PR-AUC: {val_pr_auc:.4f} | "
      f"Pos Dice: {val_pos_dice:.4f}{saved_flag}"
    )

    if no_improve_epochs >= args.patience:
      print(f"\n[EARLY STOPPING] Val ROC-AUC did not improve for {args.patience} epochs. Stopping early.")
      break

  pd.DataFrame(history).to_csv(history_path, index=False)
  print("-" * 90)
  print(f"[FINISH Stage {args.stage}] Best Val ROC-AUC: {best_roc_auc:.4f}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train Hybrid Segmentation + DAM CXR Triage Model")
  parser.add_argument("--stage", type=int, choices=[1, 2], required=True, help="1: Multi-task Pretrain, 2: DAM AUC Margin Loss")
  parser.add_argument("--model_name", type=str, default="densenet121", help="Model architecture from timm")
  parser.add_argument("--img_size", type=int, default=256, help="Input image size")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 is optimal for MPS memory)")
  parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
  parser.add_argument("--lr", type=float, default=None, help="Initial LR (default: 3e-4 for Stage 1, 0.01 for Stage 2)")
  parser.add_argument("--margin", type=float, default=0.8, help="Margin m for AUCMLoss")
  parser.add_argument("--gamma", type=float, default=500, help="Gamma regularizer for PESG")
  parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout rate")
  parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--patience", type=int, default=5, help="Early stopping patience epochs")

  args = parser.parse_args()
  main(args)