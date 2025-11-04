# model_manager.py
import base64
import io
import json
from pathlib import Path
from PIL import Image
import torch
from sympy import false
from torchvision import transforms, models
from ultralytics import YOLO
from torch import nn

from cos_access import upload_image_to_cos

# ---------------- CIFAR-10 ResNet ----------------
CIFAR_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ---------------- 读取ImageNet JSON标签 ----------------
# 确保 imagenet-simple-labels.json 与 model_manager.py 在同一目录
with open("imagenet-simple-labels.json", "r", encoding="utf-8") as f:
    IMAGENET_CLASSES = json.load(f)  # 直接解析JSON文件获取1000类标签

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3,16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1,num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.fc(x)

# ---------------- 模型管理 ----------------
class ModelManager:
    def __init__(self):
        self.MODEL_DICT = {}
        self._load_models()

    def _load_models(self):
        # 1️⃣ CIFAR-10 ResNet20
        cifar20 = ResNet(BasicBlock, [3,3,3], num_classes=10)
        cifar20.load_state_dict(torch.load("./pretrained_weights/CIFAR_ResNet/cifar10_resnet20.pth", map_location='cpu'))
        cifar20.eval()
        cifar20.task = "classification"
        cifar20.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914,0.4822,0.4465], [0.2023,0.1994,0.2010])
        ])
        cifar20.classes = CIFAR_CLASSES
        self.MODEL_DICT["cifar10_resnet20"] = cifar20

        # 2️⃣ ImageNet ResNet50
        imagenet50 = models.resnet50(pretrained=True)
        imagenet50.eval()
        imagenet50.task = "classification"
        imagenet50.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        imagenet50.classes = IMAGENET_CLASSES  # 关键：绑定ImageNet标签列表
        self.MODEL_DICT["imagenet_resnet50"] = imagenet50

        # 3️⃣ YOLOv5s
        yolov5s = YOLO("./pretrained_weights/YOLOv5/yolov5s.pt")
        # yolov5s.task = "detection"
        self.MODEL_DICT["yolov5s"] = yolov5s

        # 4️⃣ YOLOv5m
        yolov5m = YOLO("./pretrained_weights/YOLOv5/yolov5m.pt")
        # yolov5m.task = "detection"
        self.MODEL_DICT["yolov5m"] = yolov5m

        # 5️⃣ YOLOv8n
        yolov8n = YOLO("./pretrained_weights/YOLOv8/yolov8n.pt")
        # yolov8n.task = "detection"
        self.MODEL_DICT["yolov8n"] = yolov8n

    def get_model(self, model_name):
        return self.MODEL_DICT.get(model_name, None)

    def infer_classification(self, model, img: Image.Image):
        transform = getattr(model, "transform", None)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, 1)
        pred_idx = int(top_idx.item())
        pred_prob = float(top_prob.item())
        pred_class = str(pred_idx)
        if hasattr(model, "classes") and model.classes:
            pred_class = model.classes[pred_idx]
        return {"pred_idx": pred_idx, "pred_class": pred_class, "pred_confidence": pred_prob}

    def infer_detection(self, model, img: Image.Image):
        results = model.predict(source=img, verbose=False)
        detections = []

        # YOLOv5/8 都支持 results[0].boxes
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            class_idxs = r.boxes.cls.cpu().numpy().astype(int)
            class_names = [r.names[i] for i in class_idxs]

            for box, conf, cls_idx, cls_name in zip(boxes, scores, class_idxs, class_names):
                x1, y1, x2, y2 = box
                detections.append({
                    "pred_idx": int(cls_idx),
                    "pred_class": cls_name,
                    "pred_confidence": float(conf),
                    "pred_box": [float(x1), float(y1), float(x2), float(y2)]
                })

        # 生成带检测框的图片（直接得到 PIL.Image 对象）
        annotated = results[0].plot()  # 得到 numpy 数组
        annotated_img = Image.fromarray(annotated)  # 转换为 PIL.Image


        image_url = upload_image_to_cos(annotated_img)


        return {"detections": detections, "image_url":image_url}




# if __name__ == "__main__":
#     manager = ModelManager()
#     img = Image.open("test.jpg")  # 读取本地图片
#     result = manager.infer_detection(manager.MODEL_DICT["yolov5s"], img)
#     print("图片 URL:", result["image_url"])



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
# 导入COS上传from qcloud_cos.cos_config import CosConfig
from qcloud_cos import CosConfig, CosS3Client, CosClientError, CosServiceError


# ---------------- 配置常量 ----------------
COS_SECRET_ID = "AKIDfDZjgOr8B0d3GEyXwDH6h5FAqJYZP2Se"
COS_SECRET_KEY = "1oeIiuvlQr45tTpzwfAj7IFzZzB8rQoz"
COS_REGION = "ap-nanjing"
COS_BUCKET = "auto-cv-lab-1320891039"
TRAINED_MODEL_DIR = "./trained_models"  # 本地模型保存目录
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)


class TrainManager:
    def __init__(self):
        self.supported_models = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "mobilenet_v2": models.mobilenet_v2
        }
        self.supported_datasets = {
            "cifar10": datasets.CIFAR10,
            "mnist": datasets.MNIST
        }
        self.train_tasks = {}  # 存储任务状态：{task_id: {"status": "running", "result": None}}

    def _get_dataset(self, dataset_name, batch_size=32):
        """加载数据集并返回DataLoader"""
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"不支持的数据集：{dataset_name}，可选：{list(self.supported_datasets.keys())}")

        # 数据预处理（根据数据集适配）
        if dataset_name in ["cifar10"]:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            num_classes = 10
        elif dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            num_classes = 10

        # 加载训练集和验证集
        train_dataset = self.supported_datasets[dataset_name](
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = self.supported_datasets[dataset_name](
            root="./data", train=False, transform=transform, download=True
        )

        return {
            "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            "val_loader": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            "num_classes": num_classes
        }

    def _get_model(self, model_name, num_classes):
        """加载模型并修改输出层"""
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型：{model_name}，可选：{list(self.supported_models.keys())}")

        model = self.supported_models[model_name](pretrained=false)
        # 修改最后一层以适配数据集类别数
        if model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "mobilenet_v2":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    def _train_epoch(self, model, train_loader, criterion, optimizer, device):
        """单轮训练"""
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
        """验证"""
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
        """生成loss和acc曲线，返回PIL Image"""
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

        return loss_img, acc_img

    def _upload_image(self, img):
        """上传图片到COS，返回URL"""
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
        task_id = str(uuid.uuid4())[:8]  # 生成短任务ID
        self.train_tasks[task_id] = {"status": "running", "result": None}

        # 异步执行训练（实际应用中用Celery/RQ，这里简化为线程）
        import threading
        def _train_thread():
            try:
                # 1. 初始化设备、数据、模型
                device = "cuda" if torch.cuda.is_available() else "cpu"
                data = self._get_dataset(dataset_name, batch_size)
                model = self._get_model(model_name, data["num_classes"]).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # 2. 训练日志
                logs = {
                    "epochs": [],
                    "train_loss": [],
                    "train_acc": [],
                    "val_loss": [],
                    "val_acc": []
                }

                # 3. 训练循环
                for epoch in range(epochs):
                    train_loss, train_acc = self._train_epoch(model, data["train_loader"], criterion, optimizer, device)
                    val_loss, val_acc = self._validate(model, data["val_loader"], criterion, device)
                    logs["epochs"].append(epoch + 1)
                    logs["train_loss"].append(round(train_loss, 4))
                    logs["train_acc"].append(round(train_acc, 4))
                    logs["val_loss"].append(round(val_loss, 4))
                    logs["val_acc"].append(round(val_acc, 4))
                    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

                # 4. 保存模型
                model_path = os.path.join(TRAINED_MODEL_DIR, f"{model_name}_{dataset_name}_{int(time.time())}.pth")
                torch.save(model.state_dict(), model_path)

                # 5. 生成并上传曲线
                loss_img, acc_img = self._plot_curves(logs)
                loss_url = self._upload_image(loss_img)
                acc_url = self._upload_image(acc_img)

                # 6. 保存结果
                self.train_tasks[task_id] = {
                    "status": "completed",
                    "result": {
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
                self.train_tasks[task_id] = {"status": "failed", "result": f"训练失败：{str(e)}"}

        threading.Thread(target=_train_thread).start()
        return task_id

    def get_train_result(self, task_id):
        """查询训练结果"""
        if task_id not in self.train_tasks:
            return {"status": "invalid", "msg": "任务ID不存在"}
        return self.train_tasks[task_id]




### 二、应用调用：`app.py`

from model_manager import ModelManager
import time


if __name__ == "__main__":
    train_manager = TrainManager()
    # 1. 发起训练任务
    task_id = train_manager.start_train(
        model_name="resnet18",
        dataset_name="cifar10",
        epochs=2,
        batch_size=32,
        lr=0.001
    )
    print(f"训练任务已启动，任务ID：{task_id}，请等待训练完成...")

    # 2. 轮询查询结果（实际应用中可用前端轮询）
    while True:
        result = train_manager.get_train_result(task_id)
        if result["status"] == "running":
            print("训练中...")
            time.sleep(10)  # 每10秒查一次
        else:
            break

    # 3. 输出训练结果
    print("\n训练结果：")
    if result["status"] == "completed":
        res = result["result"]
        print(f"模型：{res['model_name']}，数据集：{res['dataset_name']}")
        print(f"最终验证准确率：{res['final_val_acc'] * 100:.2f}%")
        print(f"Loss曲线：{res['loss_curve_url']}")
        print(f"Acc曲线：{res['acc_curve_url']}")
    else:
        print(result["result"])