# train_manager.py (改造后)

import os
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import io
from threading import Lock
from qcloud_cos import CosConfig, CosS3Client

# ---------------- 配置常量 ----------------
COS_SECRET_ID = "AKIDfDZjgOr8B0d3GEyXwDH6h5FAqJYZP2Se"
COS_SECRET_KEY = "1oeIiuvlQr45tTpzwfAj7IFzZzB8rQoz"
COS_REGION = "ap-nanjing"
COS_BUCKET = "auto-cv-lab-1320891039"
TRAINED_MODEL_DIR = "./trained_models"
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

# ---------------- 从 model_manager 导入共享资源 ----------------
from model_manager import BasicBlock, ResNet, CIFAR_CLASSES, CIFAR100_CLASSES


class TrainManager:
    def __init__(self):
        # 1. 更新支持的模型：加入 resnet20
        self.supported_models = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet20": lambda: ResNet(BasicBlock, [3, 3, 3]),  # 复用 ModelManager 的 ResNet20 定义
            "mobilenet_v2": models.mobilenet_v2
        }

        # 2. 更新数据集配置：统一管理，复用预处理和类别
        self.dataset_config = {
            "cifar10": {
                "transform": transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
                "val_transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
                "classes": CIFAR_CLASSES,
                "num_classes": 10,
                "dataset": datasets.CIFAR10
            },
            "cifar100": {
                "transform": transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]),
                "val_transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]),
                "classes": CIFAR100_CLASSES,
                "num_classes": 100,
                "dataset": datasets.CIFAR100
            },
            "mnist": {
                # ... (保持原有配置)
            }
        }
        self.train_tasks = {}
        self.task_lock = Lock()

    def _get_dataset(self, dataset_name, batch_size=32):
        """加载数据集并返回DataLoader"""
        if dataset_name not in self.dataset_config:
            raise ValueError(f"不支持的数据集：{dataset_name}，可选：{list(self.dataset_config.keys())}")

        config = self.dataset_config[dataset_name]

        train_dataset = config["dataset"](
            root="./data", train=True, transform=config["transform"], download=True
        )
        val_dataset = config["dataset"](
            root="./data", train=False, transform=config["val_transform"], download=True
        )

        return {
            "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
            "val_loader": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
            "num_classes": config["num_classes"],
            "classes": config["classes"]
        }

    def _get_model(self, model_name, num_classes):
        """加载模型并修改输出层以适配数据集"""
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型：{model_name}，可选：{list(self.supported_models.keys())}")

        model = self.supported_models[model_name]()

        # 3. 统一修改最后一层
        if isinstance(model, ResNet):  # 针对我们自定义的 ResNet20
            # 假设 ResNet 的最后一层是 fc
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name.startswith("resnet"):  # 针对 torchvision 的 ResNet18/50
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "mobilenet_v2":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

        return model

    # ... (其余 _train_epoch, _validate, _plot_curves, _upload_image 方法保持不变)
    def _train_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        return avg_loss, acc

    def _validate(self, model, val_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        acc = correct / total
        return avg_loss, acc

    def _plot_curves(self, logs):
        plt.switch_backend('Agg')  # 非交互式后端，适合服务器环境
        # Loss曲线
        plt.figure(figsize=(10, 4))
        plt.plot(logs["epochs"], logs["train_loss"], label="Train Loss")
        plt.plot(logs["epochs"], logs["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        loss_buf = io.BytesIO()
        plt.savefig(loss_buf, format="png", bbox_inches="tight")
        loss_buf.seek(0)
        loss_img = Image.open(loss_buf)

        # Acc曲线
        plt.figure(figsize=(10, 4))
        plt.plot(logs["epochs"], logs["train_acc"], label="Train Acc")
        plt.plot(logs["epochs"], logs["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        acc_buf = io.BytesIO()
        plt.savefig(acc_buf, format="png", bbox_inches="tight")
        acc_buf.seek(0)
        acc_img = Image.open(acc_buf)

        plt.close('all')  # 关闭图表，释放内存
        return loss_img, acc_img

    def _upload_image(self, img):
        temp_path = f"temp_{uuid.uuid4()}.png"
        try:
            img.save(temp_path, format="PNG")
            config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY)
            client = CosS3Client(config)
            cos_key = f"train_plots/{uuid.uuid4()}.png"
            client.upload_file(Bucket=COS_BUCKET, LocalFilePath=temp_path, Key=cos_key)
            return f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com/{cos_key}"
        except Exception as e:
            print(f"图片上传失败：{e}")
            return ""
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def start_train(self, model_name, dataset_name, epochs=10, batch_size=32, lr=0.001):
        """启动训练任务，返回任务ID"""
        task_id = str(uuid.uuid4())[:8]
        with self.task_lock:
            self.train_tasks[task_id] = {"status": "running", "result": None}

        import threading
        def _train_thread():
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"使用设备: {device}")

                # 加载数据
                data = self._get_dataset(dataset_name, batch_size)
                train_loader, val_loader = data["train_loader"], data["val_loader"]
                num_classes = data["num_classes"]
                classes = data["classes"]

                # 加载并修改模型
                model = self._get_model(model_name, num_classes).to(device)

                # 定义损失函数和优化器
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # 可选：添加学习率调度器
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                logs = {
                    "epochs": [], "train_loss": [], "train_acc": [],
                    "val_loss": [], "val_acc": []
                }

                for epoch in range(epochs):
                    print(f"----- Epoch {epoch + 1}/{epochs} -----")
                    train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc = self._validate(model, val_loader, criterion, device)
                    scheduler.step()  # 更新学习率

                    logs["epochs"].append(epoch + 1)
                    logs["train_loss"].append(round(train_loss, 4))
                    logs["train_acc"].append(round(train_acc, 4))
                    logs["val_loss"].append(round(val_loss, 4))
                    logs["val_acc"].append(round(val_acc, 4))
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # 保存模型
                model_filename = f"{model_name}_{dataset_name}_{int(time.time())}.pth"
                model_path = os.path.join(TRAINED_MODEL_DIR, model_filename)
                # 保存整个模型状态，包括结构和参数
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'num_classes': num_classes,
                    'classes': classes
                }, model_path)

                # 生成并上传曲线
                loss_img, acc_img = self._plot_curves(logs)
                loss_url = self._upload_image(loss_img)
                acc_url = self._upload_image(acc_img)

                with self.task_lock:
                    self.train_tasks[task_id] = {
                        "status": "completed",
                        "result": {
                            "task_id": task_id,
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "epochs": epochs,
                            "final_val_acc": logs["val_acc"][-1],
                            "model_path": model_path,
                            "loss_curve_url": loss_url,
                            "acc_curve_url": acc_url,
                            "logs": logs
                        }
                    }
            except Exception as e:
                import traceback
                error_msg = f"训练失败：{str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                with self.task_lock:
                    self.train_tasks[task_id] = {"status": "failed", "result": error_msg}

        threading.Thread(target=_train_thread).start()
        return task_id

    def get_train_result(self, task_id):
        """查询训练结果"""
        with self.task_lock:
            if task_id not in self.train_tasks:
                return {"status": "invalid", "msg": "任务ID不存在"}
            return self.train_tasks[task_id]


# app.py

from train_manager import TrainManager
from model_manager import ModelManager
from PIL import Image
import time

if __name__ == "__main__":
    train_manager = TrainManager()

    # 1. 发起一个训练任务，例如在 CIFAR-100 上训练 ResNet20
    print("--- 发起训练任务 ---")
    task_id = train_manager.start_train(
        model_name="resnet20",
        dataset_name="cifar100",
        epochs=5,  # 为了快速测试， epoch 设小一点
        batch_size=64,
        lr=0.001
    )
    print(f"训练任务已启动，任务ID：{task_id}，请等待训练完成...")

    # 2. 轮询查询结果
    result = None
    while True:
        result = train_manager.get_train_result(task_id)
        if result["status"] == "running":
            print("训练中...")
            time.sleep(20)
        else:
            break

    # 3. 输出训练结果
    print("\n--- 训练结果 ---")
    if result["status"] == "completed":
        res = result["result"]
        print(f"任务 ID: {res['task_id']}")
        print(f"模型: {res['model_name']}, 数据集: {res['dataset_name']}")
        print(f"最终验证准确率: {res['final_val_acc'] * 100:.2f}%")
        print(f"模型保存路径: {res['model_path']}")
        print(f"Loss曲线: {res['loss_curve_url']}")
        print(f"Acc曲线: {res['acc_curve_url']}")
    else:
        print(f"训练失败: {result['result']}")

    # 4. 使用 ModelManager 加载并推理
    print("\n--- 加载并推理 ---")
    manager = ModelManager()

    # 找到我们刚刚训练好的模型
    trained_model_name = None
    for name in manager.MODEL_DICT.keys():
        if name.startswith(f"trained_resnet20_cifar100"):
            trained_model_name = name
            break

    if trained_model_name:
        print(f"找到训练好的模型: {trained_model_name}")
        model = manager.get_model(trained_model_name)

        # 假设你有一张测试图片 'test_cifar100.jpg'
        try:
            img = Image.open("test_cifar100.jpg").convert("RGB")
            pred_result = manager.infer_classification(model, img)
            print(f"推理结果: {pred_result}")
        except FileNotFoundError:
            print("测试图片 'test_cifar100.jpg' 未找到，跳过推理示例。")
    else:
        print("未在 ModelManager 中找到训练好的模型。")