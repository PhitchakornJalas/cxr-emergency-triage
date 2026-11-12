import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
    x = torch.cat([x, skip], dim=1)
    return self.conv(x)


class CXRSegmentationTriage(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    """
    Pure Segmentation Architecture สำหรับงาน Emergency CXR Triage:
    - ทำนาย Pixel-wise Mask [B, 1, H, W] โดยตรง
    - คะแนนความเสี่ยงระดับภาพ (Image Triage Score) คำนวณจากพิกเซลที่มั่นใจสูงสุดใน Mask
    """
    super().__init__()
    self.model_name = model_name

    self.encoder = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
    )
    chs = self.encoder.feature_info.channels()

    self.dec4 = DecoderBlock(chs[4], chs[3], 256)
    self.dec3 = DecoderBlock(256, chs[2], 128)
    self.dec2 = DecoderBlock(128, chs[1], 64)
    self.dec1 = DecoderBlock(64, chs[0], 32)
    self.dropout = nn.Dropout2d(p=drop_rate)
    self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_size = x.shape[2:]
    feats = self.encoder(x)  # [f0, f1, f2, f3, f4]

    d4 = self.dec4(feats[4], feats[3])
    d3 = self.dec3(d4, feats[2])
    d2 = self.dec2(d3, feats[1])
    d1 = self.dec1(d2, feats[0])
    d1 = self.dropout(d1)

    seg_logits = F.interpolate(self.seg_head(d1), size=input_size, mode="bilinear", align_corners=False)
    return seg_logits

  @staticmethod
  def logits_to_image_prob(seg_logits: torch.Tensor, top_k: int = 16) -> torch.Tensor:
    """
    แปลง Segmentation Logits [B, 1, H, W] เป็นคะแนนความเสี่ยงระดับภาพ [B]
    โดยเฉลี่ยค่า Probability ของ Top-K พิกเซลสูงสุด (ลดผลกระทบจาก Noise จุดเดียว)
    """
    probs = torch.sigmoid(seg_logits).flatten(1)  # [B, H*W]
    topk_vals, _ = torch.topk(probs, k=top_k, dim=1)
    return topk_vals.mean(dim=1)


def build_model(model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2) -> CXRSegmentationTriage:
  return CXRSegmentationTriage(model_name=model_name, pretrained=pretrained, drop_rate=drop_rate)


if __name__ == "__main__":
  model = build_model(model_name="densenet121", pretrained=False)
  dummy_input = torch.randn(4, 3, 256, 256)
  seg_logits = model(dummy_input)
  img_probs = model.logits_to_image_prob(seg_logits)

  total_params = sum(p.numel() for p in model.parameters()) / 1e6
  print(f"[MODEL READY] Architecture : {model.model_name} (Pure Segmentation)")
  print(f"Total Parameters          : {total_params:.2f} M")
  print(f"Output Seg Logits Shape   : {seg_logits.shape}")
  print(f"Derived Image Probs Shape : {img_probs.shape}")