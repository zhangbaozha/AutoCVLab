import json
from io import BytesIO
from typing import Dict, Any

import requests
from flask import Flask, request, jsonify, current_app
from PIL import Image
from model_manager import ModelManager, TrainManager

app = Flask(__name__)
manager = ModelManager()


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
        data = request.json or request.form
        model_name = data.get("model_name", "").strip()
        dataset_name = data.get("dataset_name", "").strip()
        epochs = int(data.get("epochs", 10))
        batch_size = int(data.get("batch_size", 32))
        lr = float(data.get("lr", 0.001))

        # 参数验证
        if not model_name or not dataset_name:
            return jsonify({"code": 400, "message": "模型名称和数据集名称为必填项"}), 400
        if epochs <= 0 or batch_size <= 0 or lr <= 0:
            return jsonify({"code": 400, "message": "轮次、批次大小和学习率必须为正数"}), 400

        # 创建任务
        task_id = TrainManager.start_train(
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
        task_id = request.args.get("task_id", "").strip()
        if not task_id:
            return jsonify({"code": 400, "message": "任务ID不能为空"}), 400

        result = TrainManager.get_train_result(task_id)
        if result["status"] == "invalid":
            return jsonify({"code": 404, "message": "任务ID不存在"}), 404

        # 计算进度
        progress = _calculate_progress(result)
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
        if not task_id:
            return jsonify({"code": 400, "message": "任务ID不能为空"}), 400

        result = TrainManager.get_train_result(task_id)
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
    # 运行中：根据已完成轮次估算
    if "result" in result and "logs" in result["result"]:
        logs = result["result"]["logs"]
        total_epochs = logs.get("epochs", [0])[-1]  # 总轮次
        completed_epochs = len(logs.get("epochs", []))  # 已完成轮次
        if total_epochs > 0:
            return min(100.0, (completed_epochs / total_epochs) * 100)
    return 0.0



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6008, debug=True)
