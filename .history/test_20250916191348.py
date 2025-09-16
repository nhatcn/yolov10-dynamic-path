import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel

# Đường dẫn đến file YAML của bạn
yaml_path = r'D:\yolov10mdf\ultralytics\ultralytics\cfg\models\v10\yolov10m.yaml'
python start_train.py --model D:\yolov10mdf\ultralytics\ultralytics\cfg\models\v10\yolov10m.yaml --data_dir coco.yaml

# Load mô hình từ YAML
model = DetectionModel(cfg=yaml_path)

# Tính tổng tham số
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Tổng số tham số: {total_params:,}")
print(f"🧠 Trong đó có thể train được: {trainable_params:,}")
