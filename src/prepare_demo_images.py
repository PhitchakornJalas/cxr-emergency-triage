import argparse
import shutil
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_test_samples(num_pneumo: int = 10, num_normal: int = 10, seed: int = 42):
  test_csv = PROJECT_ROOT / "data" / "test.csv"
  if not test_csv.exists():
    raise FileNotFoundError(f"ไม่พบไฟล์ {test_csv}")

  df = pd.read_csv(test_csv)

  # สุ่มเคสเป็น Pneumothorax (label == 1) และเคสปกติ (label == 0)
  pneumo_df = df[df["label"] == 1].sample(n=min(num_pneumo, len(df[df["label"] == 1])), random_state=seed)
  normal_df = df[df["label"] == 0].sample(n=min(num_normal, len(df[df["label"] == 0])), random_state=seed)

  sample_df = pd.concat([pneumo_df, normal_df]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

  # สร้างโฟลเดอร์ปลายทางสำหรับเก็บไฟล์ทดสอบเว็บ
  out_dir = PROJECT_ROOT / "demo_test_images"
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  for idx, row in sample_df.iterrows():
    raw_rel_path = row["img_filepath"].replace("preprocess/images_256", "raw/images").lstrip("/")
    src_path = PROJECT_ROOT / raw_rel_path
    
    tag = "PNEUMO" if int(row["label"]) == 1 else "NORMAL"
    dst_path = out_dir / f"scan_{idx+1:02d}_{tag}_{src_path.name}"
    shutil.copy2(src_path, dst_path)

  print(f"[DONE] คัดลอกรูปทดสอบทั้งหมด {len(sample_df)} ภาพ (Pneumo={len(pneumo_df)}, Normal={len(normal_df)})")
  print(f"[FOLDER] -> {out_dir}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract sample PNG images from test.csv for Web App testing")
  parser.add_argument("--num_pneumo", type=int, default=15, help="จำนวนภาพคนป่วย (Pneumothorax)")
  parser.add_argument("--num_normal", type=int, default=30, help="จำนวนภาพคนปกติ (Normal)")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  extract_test_samples(args.num_pneumo, args.num_normal, args.seed)