from DataReader import DataReader
from LocalCOCO import LocalCOCO
from torchvision import transforms

import tools
import os
import torch
import cv2
import numpy as np
import json
import ffmpeg
import xml.etree.ElementTree as ET
import re

NEW_ANNO_DIR="annotations_coco"
NEW_IMG_DIR="images_coco"
DEFAULT_FFMPEG_DICT ={
    "framestep": 1,
}

class DatasetProcessor():
    """
    Process dataset to COCO format.
    Standard bboxes annotations are in the form (x_left,y_left,width,height)
    """
    def __init__(self):
        self.new_anno_dir=NEW_ANNO_DIR
        self.new_img_dir=NEW_IMG_DIR
        self.default_ffmpeg_dict = DEFAULT_FFMPEG_DICT

    @staticmethod
    def build_frames_dir(dataset_path:str):
        """
        Build images buffer for each video.
        Args:
            dataset_path: dataset path
        Returns:
            No return value.
        """
        output_root=os.path.join(dataset_path,NEW_IMG_DIR)

        if os.path.exists(output_root) and os.path.isdir(output_root):
            print("Images buffer already exists, skip building.")
            tools.clean_create_dir_files(output_root)
        else:
            os.mkdir(output_root)
            print("Image buffer created.")

    @staticmethod
    def initialization(dataset_path:str, image_refresh_tag=False, anno_refresh_tag=True):
        dataset_type=dataset_path.split("_")[-1]

        images_output_root_path = os.path.join(dataset_path, NEW_IMG_DIR)
        anno_output_root_path = os.path.join(dataset_path, NEW_ANNO_DIR)

        if image_refresh_tag:
            tools.clean_create_dir_files(images_output_root_path)
        else:
            tools.create_dir_if_not_exists(images_output_root_path)

        if anno_refresh_tag:
            tools.clean_create_dir_files(anno_output_root_path)
        else:
            tools.create_dir_if_not_exists(anno_output_root_path)

        return images_output_root_path,anno_output_root_path,dataset_type

    @staticmethod
    def build_annotation(dataset_path:str,
                         mode:str,
                         image_refresh_tag=False,
                         anno_refresh_tag=True,
                         ffmpeg_dict=None,
                         ):
        pass

    @staticmethod
    def purdue2coco(dataset_path:str, img_re_tag=False, anno_re_tag=True, ffmpeg_dict=None):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_path: dataset path
            img_re_tag: if True, process images.
            anno_re_tag: if True, process annotations.
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        images_output_root_path, anno_output_root_path,dataset_type= DatasetProcessor.initialization(dataset_path)
        videos_root_path=os.path.join(dataset_path,"Videos","Videos")
        anno_root_path=os.path.join(dataset_path,"Video_Annotation","Video_Annotation")

        _,videos_list=DataReader.get_media_list(videos_root_path)
        videos_list=sorted(videos_list,key=lambda x:int(x.split(".")[0].split("_")[-1]))
        anno_suffix="_gt.txt"

        #Start loop for each video
        coco_anno_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        img_id_count=1
        anno_id_count=1

        for video in videos_list:
            video_name=video.split(".")[0]

            video_path=os.path.join(videos_root_path,video)
            anno_path=os.path.join(anno_root_path,video_name+anno_suffix)

            images_output_dir_path=os.path.join(images_output_root_path,video_name)
            anno_output_dir_path=os.path.join(anno_output_root_path,video_name)

            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)

            video_info=tools.get_video_info(video_path)

            if img_re_tag:
                tools.video2images_ffmpeg(video_path,images_output_dir_path,**ffmpeg_dict)
            else:
                print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

            # Put all annotations in a dict
            if anno_re_tag:
                anno_all_dict={}
                anno = DataReader.get_txt_file(anno_path)
                for line in anno:
                    frame = int(re.search(r"time_layer: (\d+)", line).group(1))
                    # Get all bounding boxes in the line
                    detections = re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", line)
                    detections = [tuple(map(int, box)) for box in detections]
                    anno_all_dict[frame]=detections

                for idx, img in enumerate(sorted(os.listdir(images_output_dir_path), key=lambda x: int(x.split(".")[0]))):
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": os.path.join(video_name, img),
                        "width": video_info["width"],
                        "height": video_info["height"],
                        "frame_id": int(img.split(".")[0]),
                        "video_id": video_name,
                        "data_type": dataset_type
                    }
                    coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 1,
                            "bbox":list(bboxes) ,
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        coco_anno_dict["annotations"].append(anno_dict)

                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(coco_anno_dict, f, indent=4)
            else:
                print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
            # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)


    @staticmethod
    def third_anti_uav2coco(dataset_path:str):
        pass

    @staticmethod
    def usc2coco(dataset_path:str):
        pass

if __name__ == '__main__':
    dataset_path="/home/king/PycharmProjects/DataMerger/Data/PURDUE_rgb"
    DatasetProcessor.purdue2coco(dataset_path,False,True)