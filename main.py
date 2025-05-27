from DataProcessor import DataProcessor
from COCO2YOLO import convert_coco_to_yolo
from ultralytics import YOLO

processor = DataProcessor()
processor.fetch_all_data_info()
outpu_put_json=processor.all_dataset_summarization()
convert_coco_to_yolo(
    json_path=f"/home/king/PycharmProjects/DataMerger/Data/all_results/{outpu_put_json}",
    output_dir="DataYOLO"
)
model = YOLO("Weights/yolov10x.pt")
# 训练配置
results = model.train(
    data="/home/king/PycharmProjects/DataMerger/DataYOLO/data.yaml",
    epochs=200,
    batch=32,
    imgsz=640,
    device="0,1",  # GPU ID
    workers=8,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.05,
)