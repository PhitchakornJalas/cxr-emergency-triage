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

def get_transforms(img_size: int = 384):
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
  def __init__(self, csv_path: Path | str, root_dir: Path = PROJECT_ROOT, transform=None):
    """
    Args:
        csv_path: Path ไปยังไฟล์ train.csv, val.csv หรือ test.csv
        root_dir: โฟลเดอร์ Root ของโปรเจกต์
        transform: Albumentations transform pipeline
    """
    self.df = pd.read_csv(csv_path).reset_index(drop=True)
    self.root_dir = Path(root_dir)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    img_path = self.root_dir / row["img_filepath"]
    mask_path = self.root_dir / row["seg_filepath"]

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
      raise FileNotFoundError(f"Cannot read mask at {mask_path}")
    mask = (mask > 0).astype("float32")

    if self.transform:
      augmented = self.transform(image=image, mask=mask)
      X = augmented["image"]
      mask_tensor = augmented["mask"].unsqueeze(0).float()  # [1, H, W]
    else:
      X = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
      mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

    cls_val = 1.0 if (mask_tensor.sum() > 0 and row["label"] == 1) else 0.0
    y = {
      "label": torch.tensor(cls_val, dtype=torch.float32),
      "mask": mask_tensor,
    }

    return X, y

if __name__ == "__main__":
  # ทดสอบการทำงานของ Dataset และ DataLoader
  train_csv = PROJECT_ROOT / "data" / "train.csv"
  train_tf, _ = get_transforms(img_size=256)

  dataset = CXRPneumothoraxDataset(csv_path=train_csv, transform=train_tf)
  loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

  X, y = next(iter(loader))
  labels = y["label"]
  masks = y["mask"]

  print(f"[TEST PASSED] Dataset Size: {len(dataset)}")
  print(f"Batch X (Images) Shape : {X.shape} | dtype: {X.dtype}")
  print(f"Batch y['label'] Shape : {labels.shape} | dtype: {labels.dtype}")
  print(f"Batch y['mask'] Shape  : {masks.shape} | dtype: {masks.dtype}")
  print(f"Sample Labels          : {labels[:8].tolist()}")