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
    def purdue2coco(dataset_path:str, img_tag=False, anno_tag=True, ffmpeg_dict=None):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_path: dataset path
            img_tag: if True, process images.
            anno_tag: if True, process annotations.
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = {}
        images_output_root_path, anno_output_root_path,dataset_type= DatasetProcessor.initialization(dataset_path)
        videos_root_path=os.path.join(dataset_path,"Videos","Videos")
        anno_root_path=os.path.join(dataset_path,"Video_Annotation","Video_Annotation")

        _,videos_list=DataReader.get_media_list(videos_root_path)
        videos_list=sorted(videos_list,key=lambda x:int(x.split(".")[0].split("_")[-1]))
        for video in videos_list:
            video_name=video.split(".")[0]

            video_path=os.path.join(videos_root_path,video)
            anno_path=os.path.join(anno_root_path,video_name+".txt")

            images_output_dir_path=os.path.join(images_output_root_path,video_name)
            anno_output_dir_path=os.path.join(anno_output_root_path,video_name)

            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)

            if img_tag:
                tools.video2images_ffmpeg(video_path,images_output_dir_path,**ffmpeg_dict)
            else:
                print(f"Images in {dataset_path} already exists, skip processing.")

            # anno=DataReader.get_txt_file(anno_path)




    @staticmethod
    def third_anti_uav2coco(dataset_path:str):
        pass

    @staticmethod
    def usc2coco(dataset_path:str):
        pass

if __name__ == '__main__':
    dataset_path="/home/king/PycharmProjects/DataMerger/Data/PURDUE_rgb"
    DatasetProcessor.purdue2coco(dataset_path)