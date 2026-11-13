import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
  """กรอง Skip Connection ด้วย Gate Signal จากชั้นลึก เพื่อลดการระบายมั่วบนกระดูก/ขอบปอด"""
  def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
    super().__init__()
    self.W_g = nn.Sequential(
      nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
      nn.BatchNorm2d(inter_ch),
    )
    self.W_x = nn.Sequential(
      nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
      nn.BatchNorm2d(inter_ch),
    )
    self.psi = nn.Sequential(
      nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
      nn.Sigmoid(),
    )
    self.relu = nn.ReLU(inplace=True)

  def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    if g1.shape[2:] != x1.shape[2:]:
        g1 = F.interpolate(g1, size=x1.shape[2:], mode="nearest")
    alpha = self.psi(self.relu(g1 + x1))
    return x * alpha


class AttentionDecoderBlock(nn.Module):
  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="nearest")  # Native MPS GPU 100%
    self.attn = AttentionGate(g_ch=in_ch, x_ch=skip_ch, inter_ch=max(skip_ch // 2, 16))
    self.conv = nn.Sequential(
        nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    x_up = self.up(x)
    if x_up.shape[2:] != skip.shape[2:]:
        x_up = F.interpolate(x_up, size=skip.shape[2:], mode="nearest")
    skip_filtered = self.attn(x_up, skip)
    return self.conv(torch.cat([x_up, skip_filtered], dim=1))


class CXRSegmentationTriage(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    super().__init__()
    self.model_name = model_name

    self.encoder = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
    )
    chs = self.encoder.feature_info.channels()  # DenseNet121: [64, 256, 512, 1024, 1024]

    self.dec4 = AttentionDecoderBlock(chs[4], chs[3], 128)
    self.dec3 = AttentionDecoderBlock(128, chs[2], 64)
    self.dec2 = AttentionDecoderBlock(64, chs[1], 32)
    self.dec1 = AttentionDecoderBlock(32, chs[0], 16)

    self.dropout = nn.Dropout2d(p=drop_rate)
    self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    feats = self.encoder(x)  # [f0, f1, f2, f3, f4]

    d4 = self.dec4(feats[4], feats[3])
    d3 = self.dec3(d4, feats[2])
    d2 = self.dec2(d3, feats[1])
    d1 = self.dec1(d2, feats[0])
    d1 = self.dropout(d1)

    seg_logits = F.interpolate(self.seg_head(d1), size=x.shape[2:], mode="nearest")
    return seg_logits

  @staticmethod
  def logits_to_image_prob(seg_logits: torch.Tensor, top_k: int = 128) -> torch.Tensor:
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
  print(f"[MODEL READY] Architecture : {model.model_name} (Attention UNet Segmentation)")
  print(f"Total Parameters          : {total_params:.2f} M")
  print(f"Output Seg Logits Shape   : {seg_logits.shape}")
  print(f"Derived Image Probs Shape : {img_probs.shape}")