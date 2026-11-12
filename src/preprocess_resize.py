from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_IMG_DIR = DATA_DIR / "processed" / "images_256"
OUT_SEG_DIR = DATA_DIR / "processed" / "segmentations_256"
IMG_SIZE = 256


def resize_pair(row_tuple):
  img_rel, seg_rel = row_tuple
  src_img = PROJECT_ROOT / img_rel
  src_seg = PROJECT_ROOT / seg_rel

  dst_img = OUT_IMG_DIR / src_img.name
  dst_seg = OUT_SEG_DIR / src_seg.name

  if not dst_img.exists():
    img = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
    if img is not None:
      img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
      cv2.imwrite(str(dst_img), img_resized)

  if not dst_seg.exists():
    seg = cv2.imread(str(src_seg), cv2.IMREAD_GRAYSCALE)
    if seg is not None:
      seg_resized = cv2.resize(seg, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
      cv2.imwrite(str(dst_seg), seg_resized)


def main():
  OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
  OUT_SEG_DIR.mkdir(parents=True, exist_ok=True)

  registry_df = pd.read_csv(DATA_DIR / "registry.csv")
  pairs = list(zip(registry_df["img_filepath"], registry_df["seg_filepath"]))

  print(f"[INFO] Pre-resizing {len(pairs)} image & mask pairs to {IMG_SIZE}x{IMG_SIZE}...")
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(resize_pair, pairs), total=len(pairs)))

  for split in ["train", "val", "test"]:
    csv_path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)
    df["img_filepath"] = df["img_filepath"].apply(
      lambda p: f"data/processed/images_256/{Path(p).name}"
    )
    df["seg_filepath"] = df["seg_filepath"].apply(
      lambda p: f"data/processed/segmentations_256/{Path(p).name}"
    )
    df.to_csv(csv_path, index=False)

  print(f"[SUCCESS] Resized images -> {OUT_IMG_DIR}")
  print(f"[SUCCESS] Resized masks  -> {OUT_SEG_DIR}")
  print("[SUCCESS] Updated train.csv, val.csv, test.csv paths!")


if __name__ == "__main__":
  main()