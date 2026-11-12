import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastDecoderBlock(nn.Module):
  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="nearest")  # Native MPS GPU 100%
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


class CXRSegmentationTriage(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    super().__init__()
    self.model_name = model_name

    self.encoder = timm.create_model(
      model_name,
      pretrained=pretrained,
      features_only=True,
      out_indices=(1, 2, 3, 4),
    )
    chs = self.encoder.feature_info.channels()  # [256, 512, 1024, 1024]

    self.dec3 = FastDecoderBlock(chs[3], chs[2], 128)  # 8x8 -> 16x16
    self.dec2 = FastDecoderBlock(128, chs[1], 64)      # 16x16 -> 32x32
    self.dec1 = FastDecoderBlock(64, chs[0], 32)       # 32x32 -> 64x64

    self.dropout = nn.Dropout2d(p=drop_rate)
    self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.encoder(x)  # [f1 (64x64), f2 (32x32), f3 (16x16), f4 (8x8)]

    d3 = self.dec3(feats[3], feats[2])
    d2 = self.dec2(d3, feats[1])
    d1 = self.dec1(d2, feats[0])
    d1 = self.dropout(d1)

    # Output ขนาด [B, 1, 64, 64] แล้วค่อยขยายเป็น [B, 1, 256, 256] ด้วย nearest
    logits_64 = self.seg_head(d1)
    seg_logits = F.interpolate(logits_64, size=x.shape[2:], mode="nearest")
    return seg_logits

  @staticmethod
  def logits_to_image_prob(seg_logits: torch.Tensor, top_k: int = 16) -> torch.Tensor:
    probs = torch.sigmoid(seg_logits).flatten(1)
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
  print(f"[MODEL READY] Architecture : {model.model_name} (Segmentation)")
  print(f"Total Parameters          : {total_params:.2f} M")
  print(f"Output Seg Logits Shape   : {seg_logits.shape}")
  print(f"Derived Image Probs Shape : {img_probs.shape}")