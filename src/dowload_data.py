import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_NAME = "vaillant/siim-ptx-png"

def download_and_extract():
  TARGET_DIR.mkdir(parents=True, exist_ok=True)
  print(f"[DIR] Target directory ready: {TARGET_DIR}")

  print(f"[DOWNLOAD] Downloading {DATASET_NAME} ...")
  cmd = [
    "kaggle", "datasets", "download",
    DATASET_NAME,
    "-p", str(TARGET_DIR)
  ]
  
  try:
    subprocess.run(cmd, check=True)
    print("[DOWNLOAD] Completed successfully.")
  except subprocess.CalledProcessError as e:
    print(f"[ERROR] Failed to download dataset: {e}")
    return
  except FileNotFoundError:
    print("[ERROR] 'kaggle' CLI not found. Make sure kaggle is installed and added to PATH.")
    return

  zip_files = list(TARGET_DIR.glob("*.zip"))
  if not zip_files:
    print("[WARN] No zip file found in target directory.")
    return

  for zip_path in zip_files:
    print(f"[UNZIP] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print(f"[UNZIP] Done. Removing archive {zip_path.name} to save disk space ...")
    zip_path.unlink()    

  subprocess.run(["rm", "-f", "../data/raw/convert_to_png.py"], check=True)
  subprocess.run(["rm", "-f", "../data/raw/mask_functions.py"], check=True)

  print(f"\n[FINISH] Dataset is ready at: {TARGET_DIR}")

if __name__ == "__main__":
  download_and_extract()