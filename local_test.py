from ultralytics.data.converter import convert_coco

# 转换 COCO 标注到 YOLO 格式
convert_coco(
    labels_dir="/home/king/PycharmProjects/DataMerger/Data/all_results/dataset_summary_2.json",  # COCO JSON 文件路径
    save_dir="yolo_labels",                 # 输出 YOLO 格式标签目录
    use_segments=False,                     # 是否转换实例分割（默认 False）
)


