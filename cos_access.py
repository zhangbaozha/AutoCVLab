import os
import uuid
from PIL import Image
from qcloud_cos import CosConfig, CosS3Client, CosClientError, CosServiceError

# COS 配置
COS_SECRET_ID = "AKIDfDZjgOr8B0d3GEyXwDH6h5FAqJYZP2Se"
COS_SECRET_KEY = "1oeIiuvlQr45tTpzwfAj7IFzZzB8rQoz"
COS_REGION = "ap-nanjing"
COS_BUCKET = "auto-cv-lab-1320891039"
COS_PATH = "detection_results/"


def upload_image_to_cos(img: Image.Image) -> str:
    """
    上传 PIL.Image 到 COS，返回图片URL（失败返回空字符串）
    """
    # 临时文件路径
    temp_path = f"temp_{uuid.uuid4()}.jpg"

    try:
        # 保存图片（强制JPG格式，避免格式问题）
        img.save(temp_path, format="JPEG")

        # 初始化COS客户端
        config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID, SecretKey=COS_SECRET_KEY)
        client = CosS3Client(config)

        # 上传到COS
        cos_key = f"{COS_PATH}{uuid.uuid4()}.jpg"
        client.upload_file(Bucket=COS_BUCKET, LocalFilePath=temp_path, Key=cos_key)

        # 返回公共访问URL
        return f"https://{COS_BUCKET}.cos.{COS_REGION}.myqcloud.com/{cos_key}"

    except Exception:
        return ""  # 任何错误都返回空

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)