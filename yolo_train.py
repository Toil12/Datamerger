from ultralytics import YOLO

# 加载模型（YOLOv10 官方或自定义模型）
# model = YOLO("yolov10n.yaml")  # 从YAML构建
# 或加载预训练权重
model = YOLO("Weights/yolov10x.pt")

# 训练配置
results = model.train(
    data="/home/king/PycharmProjects/DataMerger/DataYOLO/data.yaml",
    epochs=100,
    batch=16,
    imgsz=640,
    device="0,1",  # GPU ID
    workers=8,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.05,
)