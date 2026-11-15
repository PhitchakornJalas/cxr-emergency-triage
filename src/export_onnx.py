import argparse
import json
from pathlib import Path
import torch
from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_to_onnx(model_name: str = "densenet121", stage: int = 2):
  device = torch.device("cpu")
  ckpt_path = PROJECT_ROOT / "checkpoints" / f"{model_name}_hybrid_stage{stage}_best.pth"
  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
  img_size = checkpoint.get("img_size", 256)

  model = build_model(model_name=model_name, pretrained=False, drop_rate=0.0).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  # ปิด gradient checkpointing (memory_efficient) ก่อน Export เพื่อไม่ให้กราฟ ONNX มีโหนดซ้ำซ้อน
  if hasattr(model.encoder, "set_grad_checkpointing"):
    model.encoder.set_grad_checkpointing(False)

  # ดึง Clinical Threshold ที่ Calibrate ไว้จาก Stage นั้นๆ มาเซฟคู่กัน
  metrics_path = PROJECT_ROOT / "outputs" / "evaluations" / f"{model_name}_hybrid_stage{stage}_metrics.json"
  threshold = 0.0341
  if metrics_path.exists():
    with open(metrics_path, "r") as f:
      threshold = float(json.load(f)["clinical_triage"]["threshold"])
  else:
    print(f"[WARN] Metrics JSON not found at {metrics_path}, using fallback threshold: {threshold}")

  web_assets_dir = PROJECT_ROOT / "web_app" / "assets"
  web_assets_dir.mkdir(parents=True, exist_ok=True)

  onnx_path = web_assets_dir / f"{model_name}_triage.onnx"
  dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

  # Export ออกมาทั้ง 2 หัว: cls_logits [B, 1] และ seg_logits [B, 1, H, W]
  torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_image"],
    output_names=["cls_logits", "seg_logits"],
    dynamic_axes={
      "input_image": {0: "batch_size"},
      "cls_logits": {0: "batch_size"},
      "seg_logits": {0: "batch_size"},
    },
  )

  config_path = web_assets_dir / "config.json"
  with open(config_path, "w") as f:
    json.dump({
      "model_name": model_name,
      "architecture": f"hybrid_seg_dam_stage{stage}",
      "img_size": img_size,
      "threshold": threshold,
      "top_k": 16,
      "alpha": 0.7,
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
    }, f, indent=2)

  print(f"[EXPORTED] ONNX Model  -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
  print(f"[EXPORTED] Web Config  -> {config_path} (Threshold = {threshold}, top_k = 16, alpha = 0.7)")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Export Hybrid Seg+DAM Model to ONNX for Web Inference")
  parser.add_argument("--model_name", type=str, default="densenet121")
  parser.add_argument("--stage", type=int, choices=[1, 2], default=2)
  args = parser.parse_args()
  export_to_onnx(args.model_name, args.stage)