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

# 新增：CIFAR-100 类别列表
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


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

    def _load_trained_models(self):
        """加载 TrainManager 训练并保存的模型"""
        trained_dir = Path("./trained_models")
        if not trained_dir.exists():
            print(f"未找到训练模型目录: {trained_dir}")
            return

        for weight_file in trained_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(weight_file, map_location='cpu')

                # 检查是否是我们自定义的 checkpoint 格式
                if 'model_state_dict' in checkpoint and 'model_name' in checkpoint:
                    model_name = checkpoint['model_name']
                    num_classes = checkpoint['num_classes']

                    # 根据模型名称重新构建模型
                    if model_name == "resnet20":
                        model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
                    elif model_name == "resnet50":
                        model = models.resnet50(pretrained=False)
                        model.fc = nn.Linear(model.fc.in_features, num_classes)
                    # 可以在这里为其他模型（如 resnet18, mobilenet_v2）添加逻辑
                    else:
                        print(f"不支持自动加载的训练模型类型: {model_name}")
                        continue

                    # 加载模型参数
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()

                    # 绑定任务类型、预处理和类别标签
                    model.task = "classification"

                    # 根据数据集名称设置 transform
                    if checkpoint['dataset_name'] == 'cifar10':
                        model.transform = transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                        ])
                    elif checkpoint['dataset_name'] == 'cifar100':
                        model.transform = transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])

                    model.classes = checkpoint['classes']

                    # 在 MODEL_DICT 中注册模型
                    registered_name = f"trained_{model_name}_{checkpoint['dataset_name']}_{weight_file.stem.split('_')[-1]}"
                    self.MODEL_DICT[registered_name] = model
                    print(f"成功加载训练模型：{registered_name}")
                else:
                    print(f"文件 {weight_file.name} 不是有效的训练模型 checkpoint，跳过。")
            except Exception as e:
                print(f"加载训练模型 {weight_file.name} 失败：{e}")
