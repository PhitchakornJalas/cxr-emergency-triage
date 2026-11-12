import timm
import torch
import torch.nn as nn


class CXRTriageClassifier(nn.Module):
  def __init__(self, model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2):
    """
    Wrapper สำหรับสร้างโมเดลจำแนกภาพ CXR รองรับการสลับ Backbone ผ่าน timm
    """
    super().__init__()
    self.model_name = model_name
    self.backbone = timm.create_model(
      model_name,
      pretrained=pretrained,
      num_classes=1,  
      drop_rate=drop_rate
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.backbone(x)
    return logits.squeeze(-1)

  def get_gradcam_target_layer(self) -> nn.Module:
    """
    ระบุ Convolution Layer สุดท้ายอัตโนมัติสำหรับดึง Feature Map มาทำ Grad-CAM
    """
    if "densenet" in self.model_name:
      return self.backbone.features.norm5
    elif "resnet" in self.model_name:
      return self.backbone.layer4[-1]
    elif "efficientnet" in self.model_name or "mobilenet" in self.model_name:
      return self.backbone.conv_head
    elif "convnext" in self.model_name:
      return self.backbone.stages[-1].blocks[-1]
    else:
      raise ValueError(f"Grad-CAM target layer not configured for {self.model_name}")


def build_model(model_name: str = "densenet121", pretrained: bool = True, drop_rate: float = 0.2) -> CXRTriageClassifier:
  return CXRTriageClassifier(model_name=model_name, pretrained=pretrained, drop_rate=drop_rate)


if __name__ == "__main__":
  # ทดสอบสร้าง Baseline Model (DenseNet-121) และเช็ค Output Shape
  model = build_model(model_name="densenet121", pretrained=False)
  dummy_input = torch.randn(4, 3, 256, 256)
  output = model(dummy_input)

  total_params = sum(p.numel() for p in model.parameters()) / 1e6
  print(f"[MODEL READY] Architecture : {model.model_name}")
  print(f"Total Parameters          : {total_params:.2f} M")
  print(f"Output Logits Shape       : {output.shape}")
  print(f"Grad-CAM Target Layer     : {type(model.get_gradcam_target_layer()).__name__}")