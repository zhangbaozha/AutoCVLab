import torch
from pathlib import Path
import shutil
import urllib.request

# ----------------------
# 配置下载目录
# ----------------------
DOWNLOAD_DIR = Path("./pretrained_weights")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# 清理 torch.hub 缓存
# ----------------------
def clear_torch_hub_cache():
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"[CACHE CLEARED] {cache_dir}")
    else:
        print("[CACHE] No cache to clear")

clear_torch_hub_cache()

# ----------------------
# 保存模型 state_dict 辅助函数
# ----------------------
def save_model_state(model, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[SAVED] {save_path}")

# ======================
# 1️⃣ CIFAR-10 / CIFAR-100 ResNet
# ======================
cifar_models = [
    "cifar10_resnet20", "cifar10_resnet32", "cifar10_resnet44", "cifar10_resnet56",
    "cifar100_resnet20", "cifar100_resnet32", "cifar100_resnet44", "cifar100_resnet56"
]

for model_name in cifar_models:
    print(f"[DOWNLOAD] CIFAR model {model_name}")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        model_name,
        pretrained=True,
        force_reload=True
    )
    save_model_state(model, DOWNLOAD_DIR / "CIFAR_ResNet" / f"{model_name}.pth")

# ======================
# 2️⃣ YOLOv5 下载（直接下载 .pt 文件）
# ======================
yolov5_models = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]
yolov5_dir = DOWNLOAD_DIR / "YOLOv5"
yolov5_dir.mkdir(parents=True, exist_ok=True)

for name in yolov5_models:
    url = f"https://github.com/ultralytics/yolov5/releases/download/v6.2/{name}.pt"
    dest = yolov5_dir / f"{name}.pt"
    if not dest.exists():
        print(f"[DOWNLOAD] YOLOv5 {name} from {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"[SAVED] {dest}")
    else:
        print(f"[SKIP] {dest} already exists")

# ======================
# 3️⃣ YOLOv8 下载（ultralytics API）
# ======================
try:
    from ultralytics import YOLO
    yolov8_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    yolov8_dir = DOWNLOAD_DIR / "YOLOv8"
    yolov8_dir.mkdir(parents=True, exist_ok=True)

    for name in yolov8_models:
        dest = yolov8_dir / f"{name}.pt"
        if not dest.exists():
            print(f"[DOWNLOAD] YOLOv8 {name}")
            model = YOLO(name + ".pt")  # 自动下载到默认缓存
            model.save(dest)            # 保存到指定目录
            print(f"[SAVED] {dest}")
        else:
            print(f"[SKIP] {dest} already exists")

except ImportError:
    print("[WARNING] ultralytics not installed. Skipping YOLOv8 download.")

print("[DONE] All requested weights processed.")
