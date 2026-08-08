import kagglehub
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("manishkr1754/cardekho-used-car-data")

print("Path to dataset files:", path)

dest = os.path.join(os.path.dirname(__file__), "cardekho_dataset")
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(path, dest)

print(f"Data successfully copied to: {dest}")
