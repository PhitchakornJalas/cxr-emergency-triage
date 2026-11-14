import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDecoderBlock(nn.Module):
  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="nearest")
    self.conv = nn.Sequential(
      nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x = self.up(x)
    if x.shape[2:] != skip.shape[2:]:
      x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
    x = torch.cat([x, skip], dim=1)
    return self.conv(x)


class HybridCXRSegmentationDAM(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    super().__init__()
    self.model_name = model_name

    # 1. Shared Encoder (memory_efficient=True ลด VRAM บน MPS 40-50%)
    self.encoder = timm.create_model(
      model_name,
      pretrained=pretrained,
      features_only=True,
      out_indices=(1, 2, 3, 4),
      memory_efficient=True,
    )
    chs = self.encoder.feature_info.channels()  # [256, 512, 1024, 1024]

    # 2. Branch 1: Segmentation Decoder
    self.dec3 = FastDecoderBlock(chs[3], chs[2], 128)
    self.dec2 = FastDecoderBlock(128, chs[1], 64)
    self.dec1 = FastDecoderBlock(64, chs[0], 32)
    self.seg_dropout = nn.Dropout2d(p=drop_rate)
    self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

    # 3. Branch 2: DAM Classification Head
    self.cls_dropout = nn.Dropout(p=drop_rate)
    self.classifier = nn.Linear(chs[3], 1)

  def reset_classifier_and_freeze_decoder(self):
    """เตรียมโมเดลเข้าสู่ Stage 2: รีเซ็ตหัว Classifier ใหม่ และล็อก Decoder เพื่อไม่ให้ภาพ Mask เพี้ยน"""
    in_features = self.classifier.in_features
    self.classifier = nn.Linear(in_features, 1)

    for module in [self.dec3, self.dec2, self.dec1, self.seg_head]:
      for param in module.parameters():
        param.requires_grad = False

  def forward(self, x: torch.Tensor, apply_bsn: bool = False, run_decoder: bool = True):
    feats = self.encoder(x)  # [f1, f2, f3, f4]

    # Branch 2: Classification / Triage Score (จาก Bottleneck f4)
    pooled = F.adaptive_avg_pool2d(feats[3], (1, 1)).flatten(1)
    cls_logits = self.classifier(self.cls_dropout(pooled))

    if apply_bsn:
      cls_out = F.normalize(torch.sigmoid(cls_logits), p=2, dim=0)
    else:
      cls_out = cls_logits

    # Branch 1: Segmentation Mask (ข้ามได้ตอนเทรน Stage 2 เพื่อประหยัด VRAM และเวลา)
    if not run_decoder:
      return cls_out, None

    d3 = self.dec3(feats[3], feats[2])
    d2 = self.dec2(d3, feats[1])
    d1 = self.dec1(d2, feats[0])
    logits_64 = self.seg_head(self.seg_dropout(d1))
    seg_logits = F.interpolate(logits_64, size=x.shape[2:], mode="nearest")

    return cls_out, seg_logits

  @staticmethod
  def fuse_triage_probs(cls_logits: torch.Tensor, seg_logits: torch.Tensor, top_k: int = 16, alpha: float = 0.7) -> torch.Tensor:
    """รวมคะแนนจาก DAM Head (alpha) และ Top-k ของ Segmentation Mask (1 - alpha) ตอนใช้งานจริง"""
    cls_prob = torch.sigmoid(cls_logits).squeeze(1)
    seg_probs = torch.sigmoid(seg_logits).flatten(1)
    topk_vals, _ = torch.topk(seg_probs, k=top_k, dim=1)
    seg_prob = topk_vals.mean(dim=1)
    return alpha * cls_prob + (1.0 - alpha) * seg_prob


def build_model(model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2) -> HybridCXRSegmentationDAM:
  return HybridCXRSegmentationDAM(model_name=model_name, pretrained=pretrained, drop_rate=drop_rate)


if __name__ == "__main__":
  model = build_model(model_name="densenet121", pretrained=False)
  dummy_input = torch.randn(4, 3, 256, 256)

  # 1. ทดสอบ Stage 1 (Multi-task: คืนค่าทั้ง cls_logits และ seg_logits)
  cls_logits, seg_logits = model(dummy_input, apply_bsn=False, run_decoder=True)
  fused_probs = model.fuse_triage_probs(cls_logits, seg_logits)

  # 2. ทดสอบ Stage 2 (รีเซ็ตหัว Classifier + ล็อก Decoder + เปิด BSN)
  model.reset_classifier_and_freeze_decoder()
  bsn_probs, skipped_seg = model(dummy_input, apply_bsn=True, run_decoder=False)

  total_params = sum(p.numel() for p in model.parameters()) / 1e6
  trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

  print(f"[MODEL READY] Architecture : {model.model_name} (Hybrid Seg + DAM)")
  print(f"Total Parameters          : {total_params:.2f} M (Stage 2 Trainable: {trainable_s2:.2f} M)")
  print(f"Stage 1 Cls Logits Shape  : {cls_logits.shape}")
  print(f"Stage 1 Seg Logits Shape  : {seg_logits.shape}")
  print(f"Fused Triage Probs Shape  : {fused_probs.shape}")
  print(f"Stage 2 BSN Probs Shape   : {bsn_probs.shape} | Decoder Skipped: {skipped_seg is None}")