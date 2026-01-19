import json
import time
import sys
from datetime import datetime
from io import BytesIO
from typing import Dict, Any
from pathlib import Path

import requests
from flask import Flask, request, jsonify, current_app
from PIL import Image
from model_manager import ModelManager
from train_manager import TrainManager


# ============ 日志配置 - 重定向 print ============
class PrintLogger:
    """将 print 输出重定向到文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)  # 保留控制台输出
        self.log.write(message)
        self.log.flush()  # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

sys.stdout = PrintLogger(log_file)

print(f"[{datetime.now()}] ========== 应用启动 ==========")
app = Flask(__name__)
manager = ModelManager()
trainer = TrainManager()

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form["model"].strip()
    file_json_str = request.form["file"].strip()  # 从 form 中读取 JSON 字符串
    print("model_name:",model_name)
    print()
    print("file_json_str:",file_json_str)
    model = manager.get_model(model_name)
    if model is None:
        return jsonify({"error": f"Unsupported model: {model_name}"}), 400


    img = parse_file_info(file_json_str)

    if getattr(model, "task", "classification") == "classification":
        result = manager.infer_classification(model, img)
        return jsonify({"model": model_name, "task":model.task, **result})
    elif getattr(model, "task") == "detect":
        result = manager.infer_detection(model, img)
        return jsonify({"model": model_name, "task":model.task, **result})


def parse_file_info(file_json_str):
    """解析文件信息JSON，提取图片URL并下载"""
    try:
        # 解析JSON字符串
        file_info = json.loads(file_json_str)

        # 提取图片URL（根据平台返回的结构调整）
        image_url = file_info["FileList"][0]["FileURL"]

        # 下载图片（禁用SSL验证以兼容签名URL）
        response = requests.get(
            image_url,
            stream=True,
            verify=False,
            timeout=10  # 10秒超时
        )
        response.raise_for_status()  # 抛出HTTP错误（如404、500）

        # 转换为PIL Image对象
        return Image.open(BytesIO(response.content)).convert("RGB")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        app.logger.error(f"文件信息解析失败: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        app.logger.error(f"图片下载失败: {str(e)}")
        return None
# ---------------- 新增训练接口 ----------------
@app.route("/train/create", methods=["POST"])
def create_train_task():
    """创建训练任务"""
    try:
        data = request.form
        model_name = data.get("model_name", "").strip()
        dataset_name = data.get("dataset_name", "").strip()
        epochs = data.get("epochs", "").strip()
        epochs = int(epochs)
        batch_size = data.get("batch_size", "").strip()
        batch_size = int(batch_size)
        lr = float(data.get("lr", 0.001))
        print("model_name:",model_name)
        print()
        print("dataset_name:",dataset_name)
        print()
        print("epochs:",epochs)
        print()
        print("batch_size:",batch_size)

        # 参数验证
        if not model_name or not dataset_name:
            return jsonify({"code": 400, "message": "模型名称和数据集名称为必填项"}), 400
        if epochs <= 0 or batch_size <= 0 or lr <= 0:
            return jsonify({"code": 400, "message": "轮次、批次大小和学习率必须为正数"}), 400

        # 创建任务
        task_id = trainer.start_train(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        return jsonify({
            "code": 200,
            "message": "训练任务创建成功",
            "data": {"task_id": task_id, "status": "running"}
        })

    except ValueError as e:
        return jsonify({"code": 400, "message": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"创建训练任务错误: {str(e)}")
        return jsonify({"code": 500, "message": "创建任务失败"}), 500


@app.route("/train/status", methods=["GET"])
def get_train_status():
    """查询训练任务状态"""
    try:
        time.sleep(5)
        task_id = request.args.get("task_id", "").strip()
        print("查询成功:", task_id)
        print("task_id:",task_id)
        if not task_id:
            return jsonify({"code": 400, "message": "任务ID不能为空"}), 400

        result = trainer.get_train_result(task_id)
        if result["status"] == "invalid":
            return jsonify({"code": 404, "message": "任务ID不存在"}), 404

        # 计算进度
        progress = _calculate_progress(result)
        
        # 如果训练完成,打印结果摘要
        if result["status"] == "completed":
            res_data = result["result"]
            print("=" * 60)
            print(f"[训练完成] 任务ID: {task_id}")
            print(f"MODEL: {res_data['model_name']}")
            print(f"DATASET: {res_data['dataset_name']}")
            print(f"EPOCHS: {res_data['epochs']}")
            print(f"VAL_ACC: {res_data['final_val_acc']*100:.2f}%")
            print(f"LOSS CURVE: {res_data['loss_curve_url']}")
            print(f"ACC CURVE: {res_data['acc_curve_url']}")
            print("=" * 60)
        
        return jsonify({
            "code": 200,
            "message": "查询成功",
            "data": {
                "task_id": task_id,
                "status": result["status"],
                "progress": progress
            }
        })

    except Exception as e:
        current_app.logger.error(f"查询训练状态错误: {str(e)}")
        return jsonify({"code": 500, "message": "查询状态失败"}), 500


@app.route("/train/result", methods=["GET"])
def get_train_result():
    """获取训练任务结果（仅完成后有效）"""
    try:
        task_id = request.args.get("task_id", "").strip()
        print("task_id:", task_id)
        if not task_id:
            return jsonify({"code": 400, "message": "任务ID不能为空"}), 400

        result = trainer.get_train_result(task_id)
        if result["status"] == "invalid":
            return jsonify({"code": 404, "message": "任务ID不存在"}), 404
        if result["status"] == "running":
            return jsonify({"code": 400, "message": "任务仍在训练中，请稍后查询"}), 400
        if result["status"] == "failed":
            return jsonify({"code": 500, "message": result["result"]}), 500

        # 整理成功结果
        res_data = result["result"]
        return jsonify({
            "code": 200,
            "message": "训练完成",
            "data": {
                "task_id": task_id,
                "model_name": res_data["model_name"],
                "dataset_name": res_data["dataset_name"],
                "epochs": res_data["epochs"],
                "final_val_acc": round(res_data["final_val_acc"] * 100, 2),  # 百分比
                "loss_curve_url": res_data["loss_curve_url"],
                "acc_curve_url": res_data["acc_curve_url"],
                "model_path": res_data["model_path"],
                "logs": res_data["logs"]
            }
        })

    except Exception as e:
        current_app.logger.error(f"获取训练结果错误: {str(e)}")
        return jsonify({"code": 500, "message": "获取结果失败"}), 500


# ---------------- 工具函数 ----------------
def parse_file_info(file_json_str):
    """解析文件信息JSON，提取图片URL并下载"""
    try:
        file_info = json.loads(file_json_str)
        image_url = file_info["FileList"][0]["FileURL"]

        # 下载图片（禁用SSL验证以兼容签名URL）
        response = requests.get(
            image_url,
            stream=True,
            verify=False,
            timeout=10
        )
        response.raise_for_status()

        return Image.open(BytesIO(response.content)).convert("RGB")

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        current_app.logger.error(f"文件解析失败: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"图片下载失败: {str(e)}")
        return None


def _calculate_progress(result: Dict[str, Any]) -> float:
    """计算训练进度（0-100）"""
    if result["status"] == "completed":
        return 100.0
    if result["status"] == "failed":
        return 0.0
    # 关键修复：处理任务刚启动、result为None的情况
    if result["result"] is None:
        return 0.0  # 任务刚启动，进度为0
    # 运行中：根据已完成轮次估算
    logs = result["result"].get("logs", {})  # 用.get避免KeyError
    total_epochs = logs.get("epochs", [0])[-1] if logs.get("epochs") else 0
    completed_epochs = len(logs.get("epochs", []))
    if total_epochs > 0:
        return min(100.0, (completed_epochs / total_epochs) * 100)
    return 0.0



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6008, debug=False)