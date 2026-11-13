from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def get_transforms(img_size: int = 256):
  train_transform = A.Compose([
    # A.Resize(img_size, img_size),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Affine(
      scale=(0.95, 1.05),
      translate_percent=(-0.05, 0.05),
      rotate=(-10, 10),
      border_mode=cv2.BORDER_CONSTANT,
      p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
  ])

  val_test_transform = A.Compose([
    # A.Resize(img_size, img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
  ])

  return train_transform, val_test_transform

class CXRPneumothoraxDataset(Dataset):
  def __init__(self, csv_path: Path | str, root_dir: Path = PROJECT_ROOT, transform=None, cache_in_ram: bool = False):
    self.df = pd.read_csv(csv_path).reset_index(drop=True)
    self.root_dir = Path(root_dir)
    self.transform = transform
    self.cache_in_ram = cache_in_ram
    self.images = []
    self.targets = self.df["label"].values.astype(int)

    if self.cache_in_ram:
      for idx in range(len(self.df)):
        img_path = self.root_dir / self.df.iloc[idx]["img_filepath"]
        img = cv2.imread(str(img_path))
        if img is None:
          raise FileNotFoundError(f"Cannot read image at {img_path}")
        self.images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  def get_imratio(self) -> float:
    return float((self.df["label"] == 1).mean())

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int):
    if self.cache_in_ram:
      image = self.images[idx]
    else:
      img_path = self.root_dir / self.df.iloc[idx]["img_filepath"]
      image = cv2.imread(str(img_path))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transform:
      X = self.transform(image=image)["image"]
    else:
      X = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    y = torch.tensor(float(self.df.iloc[idx]["label"]), dtype=torch.float32)
    return X, y

if __name__ == "__main__":
  # ทดสอบการทำงานของ Dataset และ DataLoader
  train_csv = PROJECT_ROOT / "data" / "train.csv"
  train_tf, _ = get_transforms(img_size=256)

  dataset = CXRPneumothoraxDataset(csv_path=train_csv, transform=train_tf)
  loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

  X, y = next(iter(loader))

  print(f"[TEST PASSED] Dataset Size: {len(dataset)}")
  print(f"Batch X (Images) Shape : {X.shape} | dtype: {X.dtype}")
  print(f"Batch y['label'] Shape : {y.shape} | dtype: {y.dtype}")
  print(f"Sample Labels          : {y[:8].tolist()}")