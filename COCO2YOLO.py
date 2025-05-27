import json
import time
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert_coco_to_yolo(
        json_path: str | Path,
        output_dir: str | Path,
        val_ratio: float = 0.2,
        random_seed: int = 42
) -> None:
    """
    Convert COCO format annotations to YOLO format with train/val split.

    Args:
        json_path: Path to COCO format JSON file
        image_dir: Directory containing original images
        output_dir: Output directory for YOLO dataset
        val_ratio: Ratio of validation data (default: 0.2)
        random_seed: Random seed for reproducibility
    """
    # Convert to Path objects (Python 3.10+ type union syntax)
    json_path = Path(json_path)
    output_dir = Path(output_dir)

    # Create output directories using pathlib
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    # Load JSON with explicit encoding
    with json_path.open(encoding='utf-8') as f:
        data = json.load(f)

    # Build mappings with type hints
    image_id_to_info: Dict[int, Dict] = {img["id"]: img for img in data["images"]}
    category_id_to_idx: Dict[int, int] = {
        cat["id"]: idx for idx, cat in enumerate(data["categories"])
    }

    # Split dataset
    all_image_ids = list(image_id_to_info.keys())
    train_ids, val_ids = train_test_split(
        all_image_ids,
        test_size=val_ratio,
        random_state=random_seed
    )

    def convert_annotation(img_id: int, output_label_dir: Path) -> None:
        """Convert single image's annotations to YOLO format"""
        img_info = image_id_to_info[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        annotations = [data["annotations"][ann_id-1] for ann_id in img_info["related_anno"]]

        # Use pathlib for path handling
        label_path = output_label_dir / f"{img_id}.txt"

        with label_path.open('w') as f:
            for ann in annotations:
                match ann["bbox"]:  # Structural pattern matching
                    case [x, y, w, h]:
                        # Convert to YOLO format

                        x_center = (x + w / 2) / img_w
                        y_center = (y + h / 2) / img_h
                        w_norm = w / img_w
                        h_norm = h / img_h

                        if any(x>1 for x in [x_center, y_center, w_norm, h_norm]):
                            print(x,y,w,h,img_w,img_h)
                            raise ValueError(img_info["file_name"], " should not > 1")
                        class_id = category_id_to_idx[ann["category_id"]]
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    case _:
                        print(f"Warning: Invalid bbox format in image {img_id}")

    # Process datasets with progress bars
    for img_id, dataset_type in [
        *zip(train_ids, ["train"] * len(train_ids)),
        *zip(val_ids, ["val"] * len(val_ids))
    ]:
        img_info = image_id_to_info[img_id]
        src_img = img_info["file_name"]
        dst_img = output_dir / "images" / dataset_type / (str(img_id)+".jpg")

        # Copy image
        shutil.copy(src_img, dst_img)

        # Convert annotations
        convert_annotation(
            img_id,
            output_dir / "labels" / dataset_type
        )

    # Generate data.yaml using f-strings with proper escaping
    with (output_dir / "data.yaml").open('w') as f:
        categories = [cat["name"] for cat in data["categories"]]
        f.write(f"""\
path: {output_dir.absolute()}
train: images/train
val: images/val

names: 
    0: drone
""")

    print(f"Dataset conversion completed! Output at: {output_dir.absolute()}")


# Example usage
if __name__ == "__main__":
    start_time=time.time()
    convert_coco_to_yolo(
        json_path="/home/king/PycharmProjects/DataMerger/Data/all_results/dataset_summary_1.json",
        output_dir="DataYOLO"
    )
    end_time=time.time()
    print(f"Finished transformation from coco type to yolo in {end_time-start_time} s")