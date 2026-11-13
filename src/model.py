import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class CXRClassificationDAM(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    super().__init__()
    self.model_name = model_name
    self.net = timm.create_model(
      model_name,
      pretrained=pretrained,
      num_classes=1,
      drop_rate=drop_rate,
      memory_efficient=True,
    )

  def reset_classifier(self):
    """ใช้ใน Stage 2: รีเซ็ตหัว Classifier ใหม่แบบ Random Init ตามเปเปอร์"""
    self.net.reset_classifier(num_classes=1)

  def forward(self, x: torch.Tensor, apply_bsn: bool = False) -> torch.Tensor:
    logits = self.net(x)  # shape: [B, 1]
    if apply_bsn:
      probs = torch.sigmoid(logits)
      # Batch Score Normalization (L2 norm across mini-batch)
      return F.normalize(probs, p=2, dim=0)
    return logits

def build_model(model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2) -> CXRClassificationDAM:
  return CXRClassificationDAM(model_name=model_name, pretrained=pretrained, drop_rate=drop_rate)


if __name__ == "__main__":
  model = build_model(model_name="densenet121", pretrained=False)
  dummy_input = torch.randn(4, 3, 256, 256)

  # ทดสอบ Stage 1 (ได้ค่า Logits ดิบสำหรับ BCEWithLogitsLoss)
  logits = model(dummy_input, apply_bsn=False)

  # ทดสอบ Stage 2 (รีเซ็ตหัว Classifier + เปิด BSN สำหรับ AUCMLoss)
  model.reset_classifier()
  bsn_probs = model(dummy_input, apply_bsn=True)

  total_params = sum(p.numel() for p in model.parameters()) / 1e6
  print(f"[MODEL READY] Architecture : {model.model_name} (Classification DAM)")
  print(f"Total Parameters          : {total_params:.2f} M")
  print(f"Stage 1 Logits Shape      : {logits.shape}")
  print(f"Stage 2 BSN Probs Shape   : {bsn_probs.shape}")