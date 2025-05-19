import time

import pandas
import argparse
from jupyter_core.migrate import migrate_dir

from DataReader import DataReader
from Dataset import Dataset

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
    def output_form(detection:tuple,original_w:int,original_h:int,mode:str):
        if mode=="pixel":
            if detection[0]<1:
                return detection[0] * original_w, detection[1] * original_h, detection[2] * original_w, detection[3] * original_h
        elif mode=="ratio":
            if detection[0]>=1:
                return [detection[0]/original_w,detection[1]/original_h,detection[2]/original_w,detection[3]/original_h]
        return detection

    @staticmethod
    def out_put_initialization(dataset_object:Dataset,
                               media:str,
                               image_single_dir:bool=False,
                               media_direct_dir_path=None,
                               anno_direct_dir_path=None):
        """
        Initialize the output path and info.
        Args:
            dataset_object: dataset object
            media: media name
            image_single_dir: for single image initialization skipping creation of directory
        Returns:
            media_name: media name
            media_path: media path
            anno_path: annotation path
            images_output_dir_path: images output directory path
            anno_output_dir_path: annotation output directory path
            media_info: media info

        """
        media_name=media.split(".")[0]
        if media_direct_dir_path is not None:
            media_path = os.path.join(media_direct_dir_path, media)
            anno_path = os.path.join(anno_direct_dir_path, media_name + dataset_object.anno_suffix)
        else:
            media_path = os.path.join(dataset_object.media_root_path, media)
            anno_path = os.path.join(dataset_object.anno_root_path, media_name + dataset_object.anno_suffix)

        if image_single_dir:
            images_output_dir_path = dataset_object.images_output_root_path
            anno_output_dir_path = dataset_object.anno_output_root_path

        else:
            images_output_dir_path = os.path.join(dataset_object.images_output_root_path, media_name)
            anno_output_dir_path = os.path.join(dataset_object.anno_output_root_path, media_name)
            # tools.create_dir_if_not_exists(images_output_dir_path)
            # tools.create_dir_if_not_exists(anno_output_dir_path)



        if media.endswith(DataReader.support_video_format()):
            media_info = tools.get_video_info(media_path)
        elif media.endswith(DataReader.support_image_format()):
            media_info = tools.get_image_info(media_path)
        else:
            raise TypeError(f"{media} is not with correct suffix.")

        return media_name,media_path,anno_path,images_output_dir_path,anno_output_dir_path,media_info

    @staticmethod
    def anno_processor_entry(anno,dataset_name:str,info:dict,mode:str="pixel",**kwargs):
        """
        Entry function for annotation processor.
        Args:
            anno: annotation
            dataset_name: dataset name
            info: media info
            mode: pixel or ratio
        Returns:
            anno_dict: annotation as dictionary
        """
        if dataset_name=="purdue_rgb":
            return DatasetProcessor.purdue_rgb_anno_processor(anno,info,mode)
        elif dataset_name=="real_world_rgb":
            return DatasetProcessor.real_world_rgb_anno_processor(anno,info,mode)
        elif dataset_name=="anti_uav_rgbt_mix":
            return DatasetProcessor.anti_uav_rgbt_mix_anno_processor(anno,info,mode)
        elif dataset_name=="anti_uav410_thermal":
            return DatasetProcessor.anti_uav410_thermal_anno_processor(anno,info,mode)
        elif dataset_name=="jet_fly_rgb":
            return DatasetProcessor.jet_fly_rgb_anno_processor(anno,info,mode)
        elif dataset_name=="fdb_rgb":
            return DatasetProcessor.fdb_rgb_anno_processor(anno,info,mode)
        elif dataset_name=="youtube_rgb":
            return DatasetProcessor.youtube_rgb_anno_processor(anno,info,mode,kwargs["idx"])
        elif dataset_name=="3rd_anti_uav_thermal":
            return DatasetProcessor.third_anti_uav_thermal_anno_processor(anno,info,mode,kwargs["json_tag"])
        elif dataset_name == "uav123_rgb":
            return DatasetProcessor.uav123_rgb_anno_processor(anno, info, mode, kwargs["idx"])
        elif dataset_name == "ard_mav_rgb":
            return DatasetProcessor.ard_mav_rgb_anno_processor(anno, info, mode)
        elif dataset_name == "drone_dataset_uav_rgb":
            return DatasetProcessor.drone_dataset_uav_rgb_anno_processor(anno, info, mode, kwargs["idx"])
        elif dataset_name == "midgard_rgb":
            return DatasetProcessor.midgard_rgb_anno_processor(anno, info, mode, kwargs["idx"])
        return None

    @staticmethod
    def build_annotation(dataset_path:str,
                         mode:str,
                         image_refresh_tag=False,
                         anno_refresh_tag=True,
                         ffmpeg_dict=None,
                         ):
        pass

    @staticmethod
    def purdue_rpg_processor(dataset_object:Dataset,dataset_name,ffmpeg_dict=None,**kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        _,videos_list=DataReader.get_media_list(dataset_object.media_root_path)
        videos_list=sorted(videos_list,key=lambda x:int(x.split(".")[0].split("_")[-1]))
        #Start loop for each video
        img_id_count=1
        anno_id_count=1
        for video in videos_list:
            video_name,video_path, anno_path, images_output_dir_path, anno_output_dir_path, video_info=DatasetProcessor.out_put_initialization(dataset_object,video)
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Refresh the video if necessary
            if dataset_object.image_refresh_tag:
                tools.video2images_ffmpeg(video_path,images_output_dir_path,**ffmpeg_dict)
            else:
                print(f"Image Refresh tag is False, {dataset_path} skip image processing.")
            # Refresh the annotations if necessary
            if dataset_object.anno_refresh_tag:
                anno = DataReader.get_txt_file(anno_path)
                # Put all annotations in a dict according to the dataset name
                anno_all_dict=DatasetProcessor.anno_processor_entry(anno,dataset_name=dataset_name,info=video_info)
                # Start processing annotations
                for idx, img in enumerate(sorted(os.listdir(images_output_dir_path), key=lambda x: int(x.split(".")[0]))):
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": img,
                        "width": video_info["width"],
                        "height": video_info["height"],
                        "frame_id": int(img.split(".")[0]),
                        "video_id": video_name,
                        "data_type":dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox":list(bboxes) ,
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)

                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()
            else:
                print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")

    @staticmethod
    def purdue_rgb_anno_processor(anno:list,info:dict,mode:str)->dict:
        results={}
        for line in anno:
            # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
            frame = int(re.search(r"time_layer: (\d+)", line).group(1))
            # Get all bounding boxes in the line
            detections = re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", line)
            detections = [tuple(map(int, box)) for box in detections]
            detections = [
                DatasetProcessor.output_form((box[1], box[0], box[3] - box[1], box[2] - box[0]),info["width"],info["height"],mode) for box in detections
            ]
            results[frame] = detections
        return results

    #TODO finish real world rgb
    @staticmethod
    def real_world_rgb_processor(dataset_object: Dataset, dataset_name:str,ffmpeg_dict=None,**kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        images_list, _ = DataReader.get_media_list(dataset_object.media_root_path)
        # Start loop for each video
        img_id_count = 1
        anno_id_count = 1
        anno_output_dir_path = ""
        #TODO continue the change
        for idx,image in enumerate(images_list):
            image_name, image_path, anno_path, _, _, image_info = DatasetProcessor.out_put_initialization(
                dataset_object, image,image_single_dir=True)
            # Refresh the images if necessary
            images_output_dir_path = os.path.join(
                dataset_object.images_output_root_path,
                "DroneTestDataset"
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                "DroneTestDataset")
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            if dataset_object.image_refresh_tag:
                tools.image_copy(image_path, images_output_dir_path)
            else:
                print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

            # Refresh the annotations if necessary
            if dataset_object.anno_refresh_tag:
                anno = DataReader.get_xml_file(anno_path)
                # Put all annotations in a dict according to the dataset name
                anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,info=image_info)
                # Start processing annotations
                # Add image info to the annotations file.
                image_dict = {
                    "id": int(img_id_count),
                    "file_path": image,
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "frame_id": image.split(".")[0],
                    "video_id": -1,
                    "data_type": dataset_object.dataset_type
                }
                dataset_object.coco_anno_dict["images"].append(image_dict)
                img_id_count += 1

                if image_dict["frame_id"] not in anno_all_dict.keys():
                    continue
                for bboxes in anno_all_dict[image_dict["frame_id"]]:
                    anno_dict = {
                        "id": int(anno_id_count),
                        "image_id": image_dict["id"],
                        "category_id": 0,
                        "bbox": list(bboxes),
                        "iscrowd": 0
                    }
                    anno_id_count += 1
                    dataset_object.coco_anno_dict["annotations"].append(anno_dict)
            else:
                print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")

        # Append category information
        dataset_object.coco_anno_dict["categories"].append({
            "id": 0,
            "name": "drone",
        })
        with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
            json.dump(dataset_object.coco_anno_dict, f, indent=4)
        dataset_object.reset_coco()

    @staticmethod
    def real_world_rgb_anno_processor(anno:ET.ElementTree,info:dict,mode:str) -> dict:
        results = {}
        detections=[]
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        frame = anno.find("filename").text.split(".")[0]
        bboxes=anno.findall("object")
        for bbox in bboxes:
            xmin=bbox.find("bndbox").find("xmin").text
            ymin=bbox.find("bndbox").find("ymin").text
            xmax=bbox.find("bndbox").find("xmax").text
            ymax=bbox.find("bndbox").find("ymax").text
            d=(int(xmin),int(ymin),int(xmax)-int(xmin),int(ymax)-int(ymin))
            detections.append(
                DatasetProcessor.output_form(d,info["width"],info["height"],mode)
            )
        results[frame] = detections
        return results

    @staticmethod
    def anti_uav_rgbt_mix_processor(dataset_object: Dataset, dataset_name:str,ffmpeg_dict=None,**kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        for media_root_dir in dataset_object.media_root_path:
            media_dir_list = DataReader.get_media_dir_list(media_root_dir)

            for media_dir in media_dir_list:
                anno_dir = media_dir
                anno_direct_dir_path=os.path.join(media_root_dir,anno_dir)
                media_direct_dir_path = os.path.join(media_root_dir, media_dir)
                _, videos_list = DataReader.get_media_list(media_direct_dir_path)
                videos_list = sorted(videos_list)
                # Start loop for each video
                for video in videos_list:
                    img_id_count = 1
                    anno_id_count = 1
                    video_name, video_path, anno_path, _, _, video_info = DatasetProcessor.out_put_initialization(
                        dataset_object=dataset_object,
                        media=video,
                        media_direct_dir_path=media_direct_dir_path,
                        anno_direct_dir_path=anno_direct_dir_path)
                    images_output_dir_path=os.path.join(
                        dataset_object.images_output_root_path,
                        f"{media_dir}_{video_name}",

                    )
                    anno_output_dir_path = os.path.join(
                        dataset_object.anno_output_root_path,
                        f"{anno_dir}_{video_name}"
                    )
                    tools.create_dir_if_not_exists(images_output_dir_path)
                    tools.create_dir_if_not_exists(anno_output_dir_path)

                    # Refresh the video if necessary
                    if dataset_object.image_refresh_tag:
                        tools.video2images_ffmpeg(video_path, images_output_dir_path, **ffmpeg_dict)
                    else:
                        print(f"Image Refresh tag is False, {dataset_path} skip image processing.")
                    # Refresh the annotations if necessary
                    if dataset_object.anno_refresh_tag:
                        anno = DataReader.get_json_anno(anno_path)
                        # Put all annotations in a dict according to the dataset name
                        anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,info=video_info)
                        # Start processing annotations
                        for idx, img in enumerate(sorted(os.listdir(images_output_dir_path), key=lambda x: int(x.split(".")[0]))):
                            # Add image info to the annotations file.
                            image_dict = {
                                "id": int(img_id_count),
                                "file_path": os.path.join(images_output_dir_path, img),
                                "width": video_info["width"],
                                "height": video_info["height"],
                                "frame_id": int(img.split(".")[0]),
                                "video_id": video_name,
                                "data_type": dataset_object.dataset_type
                            }
                            dataset_object.coco_anno_dict["images"].append(image_dict)
                            img_id_count += 1
                            if image_dict["frame_id"] not in anno_all_dict.keys():
                                continue
                            # print(image_dict["frame_id"],len(anno_all_dict[image_dict["frame_id"]]))
                            for bboxes in anno_all_dict[image_dict["frame_id"]]:
                                anno_dict = {
                                    "id": int(anno_id_count),
                                    "image_id": image_dict["id"],
                                    "category_id": 0,
                                    "bbox": list(bboxes),
                                    "iscrowd": 0
                                }
                                anno_id_count += 1
                                dataset_object.coco_anno_dict["annotations"].append(anno_dict)

                        dataset_object.coco_anno_dict["categories"].append({
                            "id": 0,
                            "name": "drone",
                        })
                        with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                            json.dump(dataset_object.coco_anno_dict, f, indent=4)
                        dataset_object.reset_coco()
                    else:
                        print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")

    @staticmethod
    def anti_uav_rgbt_mix_anno_processor(anno,info:dict,mode:str="pixel") -> dict:
        results = {}

        frame_exist_list=anno["exist"]
        rects_list=anno["gt_rect"]
        for idx,existence in enumerate(frame_exist_list):
            frame=idx+1
            if not existence:
                continue
            else:
                rect = rects_list[idx]
                xmin=rect[0]
                ymin=rect[1]
                w=rect[2]
                h=rect[3]
                d=(int(xmin),int(ymin),w,h)
                results[frame] = [DatasetProcessor.output_form(d,info["width"],info["height"],mode)]
        return results

    @staticmethod
    def anti_uav410_thermal_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        for media_root_dir in dataset_object.media_root_path:
            media_dir_list = DataReader.get_media_dir_list(media_root_dir)
            # For each video
            for media_dir in media_dir_list:
                anno_dir = media_dir
                anno_direct_dir_path = os.path.join(dataset_object.anno_output_root_path,media_root_dir, anno_dir)
                media_direct_dir_path = os.path.join(dataset_object.anno_output_root_path,media_root_dir, media_dir)
                images_list, _ = DataReader.get_media_list(media_direct_dir_path)
                images_list = sorted(images_list,key=lambda x:int(x.split(".")[0]))

                # Start loop for each video
                img_id_count = 1
                anno_id_count = 1
                images_output_dir_path = os.path.join(
                    dataset_object.images_output_root_path,
                    media_dir
                )
                anno_output_dir_path = os.path.join(
                    dataset_object.anno_output_root_path,
                    media_dir)
                tools.create_dir_if_not_exists(images_output_dir_path)
                tools.create_dir_if_not_exists(anno_output_dir_path)
                # Start loop of each image
                for idx, image in enumerate(images_list):
                    image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                        dataset_object, image,
                        image_single_dir=True,
                        media_direct_dir_path=media_direct_dir_path,
                        anno_direct_dir_path=anno_direct_dir_path)
                    # Refresh the images if necessary
                    if dataset_object.image_refresh_tag:
                        tools.image_copy(image_path, images_output_dir_path)
                    else:
                        print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                    # Refresh the annotations if necessary

                    if dataset_object.anno_refresh_tag:
                        anno = DataReader.get_json_anno(os.path.join(anno_direct_dir_path,dataset_object.anno_suffix))
                        # Put all annotations in a dict according to the dataset name
                        anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                              info=image_info)
                        # Start processing annotations
                        # Add image info to the annotations file.
                        image_dict = {
                            "id": int(img_id_count),
                            "file_path": image,
                            "width": image_info["width"],
                            "height": image_info["height"],
                            "frame_id": int(image.split(".")[0]),
                            "video_id": -1,
                            "data_type": dataset_object.dataset_type
                        }
                        dataset_object.coco_anno_dict["images"].append(image_dict)
                        img_id_count += 1

                        if image_dict["frame_id"] not in anno_all_dict.keys():
                            continue
                        for bboxes in anno_all_dict[image_dict["frame_id"]]:
                            anno_dict = {
                                "id": int(anno_id_count),
                                "image_id": image_dict["id"],
                                "category_id": 0,
                                "bbox": list(bboxes),
                                "iscrowd": 0
                            }
                            anno_id_count += 1
                            dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                        else:
                            print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
                # Append category information
                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()



    @staticmethod
    def anti_uav410_thermal_anno_processor(anno, info: dict, mode: str = "pixel") -> dict:
        results = {}
        detections = []
        frame_exist_list = anno["exist"]
        rects_list = anno["gt_rect"]

        for idx, existence in enumerate(frame_exist_list):
            frame = idx + 1
            rect = rects_list[idx]
            if not existence:
                continue
            else:
                xmin = rect[0]
                ymin = rect[1]
                w = rect[2]
                h = rect[3]
                d = (int(xmin), int(ymin), w, h)
                detections.append(
                    DatasetProcessor.output_form(d, info["width"], info["height"], mode)
                )
                results[frame] = detections
        return results

    @staticmethod
    def jet_fly_rgb_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)
        for media_dir in media_dir_list:

            # For each video

            anno_dir = media_dir
            anno_direct_dir_path = os.path.join(dataset_object.dataset_path,dataset_object.anno_root_path,anno_dir)
            media_direct_dir_path = os.path.join(dataset_object.dataset_path,dataset_object.media_root_path, media_dir)
            images_list, _ = DataReader.get_media_list(media_direct_dir_path)
            images_list = sorted(images_list, key=lambda x: int(x.split(".")[0]))

            # Start loop for each video
            img_id_count = 1
            anno_id_count = 1
            images_output_dir_path = os.path.join(
                dataset_object.images_output_root_path,
                media_dir
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                media_dir)
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Start loop of each image
            for idx, image in enumerate(images_list):
                image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                    dataset_object, image,
                    image_single_dir=True,
                    media_direct_dir_path=media_direct_dir_path,
                    anno_direct_dir_path=anno_direct_dir_path)
                # Refresh the images if necessary
                if dataset_object.image_refresh_tag:
                    tools.image_copy(image_path, images_output_dir_path)
                else:
                    print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                # Refresh the annotations if necessary

                if dataset_object.anno_refresh_tag:
                    anno_path=os.path.join(anno_direct_dir_path,image_name+dataset_object.anno_suffix)
                    anno = DataReader.get_xml_file(anno_path)
                    # Put all annotations in a dict according to the dataset name
                    anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                          info=image_info)
                    # Start processing annotations
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": image,
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "frame_id": int(image.split(".")[0][2:]),
                        "video_id": -1,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                    else:
                        print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
            # Append category information
            dataset_object.coco_anno_dict["categories"].append({
                "id": 0,
                "name": "drone",
            })
            with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                json.dump(dataset_object.coco_anno_dict, f, indent=4)
            dataset_object.reset_coco()

    @staticmethod
    def jet_fly_rgb_anno_processor(anno: ET.ElementTree, info: dict, mode: str) -> dict:
        results = {}
        detections = []
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        frame = anno.find("filename").text.split(".")[0]
        bboxes = anno.findall("object")
        for bbox in bboxes:
            xmin = bbox.find("bndbox").find("xmin").text
            ymin = bbox.find("bndbox").find("ymin").text
            xmax = bbox.find("bndbox").find("xmax").text
            ymax = bbox.find("bndbox").find("ymax").text
            d = (int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin))
            detections.append(
                DatasetProcessor.output_form(d, info["width"], info["height"], mode)
            )
        results[frame] = detections
        return results

    @staticmethod
    def fdb_rgb_processor(dataset_object: Dataset, dataset_name, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        _, videos_list = DataReader.get_media_list(os.path.join(dataset_object.media_root_path,"drones"))

        videos_list = sorted(videos_list, key=lambda x: int(x.split(".")[0].split("_")[-1]))
        # Start loop for each video
        img_id_count = 1
        anno_id_count = 1
        for video in videos_list:
            media_direct_dir_path = os.path.join(dataset_object.media_root_path, "drones")
            anno_direct_dir_path = os.path.join(dataset_object.anno_root_path, "drones")
            video_name, video_path, anno_path, _, _, video_info = DatasetProcessor.out_put_initialization(
                dataset_object,
                video,
                media_direct_dir_path=media_direct_dir_path,
                anno_direct_dir_path=anno_direct_dir_path)
            images_output_dir_path=os.path.join(
                dataset_object.images_output_root_path,
                video_name
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                video_name
            )
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Refresh the video if necessary
            if dataset_object.image_refresh_tag:
                tools.video2images_ffmpeg(video_path, images_output_dir_path, **ffmpeg_dict)
            else:
                print(f"Image Refresh tag is False, {dataset_path} skip image processing.")
            # Refresh the annotations if necessary
            if dataset_object.anno_refresh_tag:
                anno = DataReader.get_txt_file(anno_path)
                # Put all annotations in a dict according to the dataset name
                anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name, info=video_info)
                # Start processing annotations
                for idx, img in enumerate(
                        sorted(os.listdir(images_output_dir_path), key=lambda x: int(x.split(".")[0]))):
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": img,
                        "width": video_info["width"],
                        "height": video_info["height"],
                        "frame_id": int(img.split(".")[0]),
                        "video_id": video_name,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)

                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()
            else:
                print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")

    @staticmethod
    def fdb_rgb_anno_processor(anno: list, info: dict, mode: str) -> dict:
        results = {}
        for line in anno:
            # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
            frame = int(re.search(r"time_layer: (\d+)", line).group(1))
            # Get all bounding boxes in the line
            detections = re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", line)
            detections = [tuple(map(int, box)) for box in detections]
            detections = [
                DatasetProcessor.output_form((box[1], box[0], box[3] - box[1], box[2] - box[0]),
                                             info["width"],
                                             info["height"],
                                             mode) for box in detections
            ]
            results[frame] = detections
        return results

    @staticmethod
    def youtube_rgb_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT


        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)

        # For each vide
        for media_dir in media_dir_list:
            anno_dir = media_dir
            anno_direct_dir_path = os.path.join(dataset_object.anno_root_path, anno_dir)
            media_direct_dir_path = os.path.join(dataset_object.media_root_path, media_dir)
            images_list, _ = DataReader.get_media_list(media_direct_dir_path)
            images_list = sorted(images_list,key=lambda x:int(x.split(".")[0]))

            # Start loop for each video
            img_id_count = 1
            anno_id_count = 1
            images_output_dir_path = os.path.join(
                dataset_object.images_output_root_path,
                media_dir
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                media_dir)
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Start loop of each image
            anno_path=os.path.join(anno_direct_dir_path,"anotation.mat")
            for idx, image in enumerate(images_list):
                image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                    dataset_object, image,
                    image_single_dir=True,
                    media_direct_dir_path=media_direct_dir_path,
                    anno_direct_dir_path=anno_direct_dir_path)
                # Refresh the images if necessary
                if dataset_object.image_refresh_tag:
                    tools.image_copy(image_path, images_output_dir_path)
                else:
                    print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                # Refresh the annotations if necessary

                if dataset_object.anno_refresh_tag:
                    anno = DataReader.get_mat_file(anno_path)
                    # Put all annotations in a dict according to the dataset name
                    anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                          info=image_info,idx=idx)
                    # Start processing annotations
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": image,
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "frame_id": int(int(image.split(".")[0])/10),
                        "video_id": -1,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                else:
                    print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
            # Append category information
            dataset_object.coco_anno_dict["categories"].append({
                "id": 0,
                "name": "drone",
            })
            with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                json.dump(dataset_object.coco_anno_dict, f, indent=4)
            dataset_object.reset_coco()

    @staticmethod
    def youtube_rgb_anno_processor(anno:dict, info: dict, mode: str, idx) -> dict:
        results = {}
        detections = []
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        rect=anno["box"][idx]
        frame=idx+1
        xmin=rect[0]
        ymin=rect[1]
        w=rect[2]
        h=rect[3]
        d=((xmin),(ymin),w,h)
        detection=[DatasetProcessor.output_form(d,info["width"],info["height"],mode)]
        results[frame] = detection

        return results

    @staticmethod
    def third_anti_uav_thermal_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT

        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)
        for media_root_dir in media_dir_list:
            media_dir_list = DataReader.get_media_dir_list(os.path.join(dataset_object.dataset_path,media_root_dir))
            # For each video
            # print(os.path.join(dataset_object.dataset_path,
            #              media_root_dir + dataset_object.anno_suffix))
            anno_path=os.path.join(dataset_object.dataset_path,media_root_dir+dataset_object.anno_suffix)
            anno = DataReader.get_json_anno(anno_path)
            for media_dir in media_dir_list:
                media_direct_dir_path = os.path.join(dataset_object.media_root_path,media_root_dir, media_dir)
                images_list, _ = DataReader.get_media_list(media_direct_dir_path)
                images_list = sorted(images_list,key=lambda x:int(x.split(".")[0]))

                # Start loop for each video
                img_id_count = 1
                anno_id_count = 1
                images_output_dir_path = os.path.join(
                    dataset_object.images_output_root_path,
                    media_dir
                )
                anno_output_dir_path = os.path.join(
                    dataset_object.anno_output_root_path,
                    media_dir)
                tools.create_dir_if_not_exists(images_output_dir_path)
                tools.create_dir_if_not_exists(anno_output_dir_path)
                # Start loop of each image
                for idx, image in enumerate(images_list):
                    image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                        dataset_object, image,
                        image_single_dir=True,
                        media_direct_dir_path=media_direct_dir_path,
                        anno_direct_dir_path="")
                    # Refresh the images if necessary
                    if dataset_object.image_refresh_tag:
                        tools.image_copy(image_path, images_output_dir_path)
                    else:
                        print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                    # Refresh the annotations if necessary
                    if dataset_object.anno_refresh_tag:
                        # Put all annotations in a dict according to the dataset name
                        json_tag=os.path.join(media_dir,image)
                        anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                              info=image_info,json_tag=json_tag)
                        # Start processing annotations
                        # Add image info to the annotations file.
                        image_dict = {
                            "id": int(img_id_count),
                            "file_path": image,
                            "width": image_info["width"],
                            "height": image_info["height"],
                            "frame_id": int(image.split(".")[0]),
                            "video_id": -1,
                            "data_type": dataset_object.dataset_type
                        }
                        dataset_object.coco_anno_dict["images"].append(image_dict)
                        img_id_count += 1

                        if image_dict["frame_id"] not in anno_all_dict.keys():
                            continue
                        for bboxes in anno_all_dict[image_dict["frame_id"]]:
                            anno_dict = {
                                "id": int(anno_id_count),
                                "image_id": image_dict["id"],
                                "category_id": 0,
                                "bbox": list(bboxes),
                                "iscrowd": 0
                            }
                            anno_id_count += 1
                            dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                    else:
                        print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
                # Append category information
                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()



    @staticmethod
    def third_anti_uav_thermal_anno_processor(anno, info: dict,mode: str = "pixel",  json_tag:str=None) -> dict:
        results = {}
        detections = []
        #
        frame=int(os.path.basename(json_tag).split(".")[0])
        img_id=-1
        for img in anno["images"]:
            if img["file_name"] == json_tag:
                img_id=img["id"]
                break

        if img_id==-1:
            raise ValueError(f"Image {json_tag} not found in annotation file.")

        search_count=0
        for a in anno["annotations"]:
            if a["image_id"]==img_id:
                rect=a["bbox"]
                xmin=rect[0]
                ymin=rect[1]
                w=rect[2]
                h=rect[3]
                d=((xmin),(ymin),w,h)
                detections.append(
                    DatasetProcessor.output_form(d, info["width"], info["height"], mode)
                )
                search_count+=1
            elif search_count>0:
                break
            else:
                pass
        results[frame] = detections
        return results

    @staticmethod
    def uav123_rgb_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT


        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)

        # For each vide
        for media_dir in media_dir_list:
            if "uav" not in media_dir or media_dir=="uav1":
                continue
            anno_dir = media_dir
            anno_direct_dir_path = dataset_object.anno_root_path
            media_direct_dir_path = os.path.join(dataset_object.media_root_path, media_dir)
            images_list, _ = DataReader.get_media_list(media_direct_dir_path)
            images_list = sorted(images_list,key=lambda x:int(x.split(".")[0]))

            # Start loop for each video
            img_id_count = 1
            anno_id_count = 1
            images_output_dir_path = os.path.join(
                dataset_object.images_output_root_path,
                media_dir
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                media_dir)

            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Start loop of each image
            anno_path = os.path.join(anno_direct_dir_path, media_dir + dataset_object.anno_suffix)
            for idx, image in enumerate(images_list):
                image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                    dataset_object, image,
                    image_single_dir=True,
                    media_direct_dir_path=media_direct_dir_path,
                    anno_direct_dir_path=anno_direct_dir_path)
                # Refresh the images if necessary
                if dataset_object.image_refresh_tag:
                    tools.image_copy(image_path, images_output_dir_path)
                else:
                    print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                # Refresh the annotations if necessary

                if dataset_object.anno_refresh_tag:
                    anno = DataReader.get_txt_file(anno_path)

                    # Put all annotations in a dict according to the dataset name
                    anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                          info=image_info,idx=idx)
                    if len(anno_all_dict.keys())==0:
                        continue
                    # Start processing annotations
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": image,
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "frame_id": int(image.split(".")[0]),
                        "video_id": -1,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                else:
                    print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
            # Append category information
            dataset_object.coco_anno_dict["categories"].append({
                "id": 0,
                "name": "drone",
            })
            with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                json.dump(dataset_object.coco_anno_dict, f, indent=4)
            dataset_object.reset_coco()

    @staticmethod
    def uav123_rgb_anno_processor(anno:dict, info: dict, mode: str, idx) -> dict:
        results = {}
        detections = []
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        frame=idx+1
        rect=anno[idx].strip().split(",")
        if "NaN" in rect:
            return results
        else:
            xmin=float(rect[0])
            ymin=float(rect[1])
            w=float(rect[2])
            h=float(rect[3])
            d=(xmin,ymin,w,h)
            detections.append(
                DatasetProcessor.output_form(d, info["width"], info["height"], mode)
            )
            results[frame] = detections

        return results

    @staticmethod
    def ard_mav_rgb_processor(dataset_object: Dataset, dataset_name, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        _, videos_list = DataReader.get_media_list(dataset_object.media_root_path)
        videos_list = sorted(videos_list, key=lambda x: int(x.split(".")[0][-2:]))
        # Start loop for each video
        img_id_count = 1
        anno_id_count = 1
        for video in videos_list:

            video_name, video_path, _, images_output_dir_path, anno_output_dir_path, video_info = DatasetProcessor.out_put_initialization(
                dataset_object, video)
            anno_dir = video_name
            anno_path = os.path.join(dataset_object.anno_root_path, anno_dir)
            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Refresh the video if necessary
            if dataset_object.image_refresh_tag:
                tools.video2images_ffmpeg(video_path, images_output_dir_path, **ffmpeg_dict)
            else:
                print(f"Image Refresh tag is False, {dataset_path} skip image processing.")
            # Refresh the annotations if necessary
            if dataset_object.anno_refresh_tag:
                anno=anno_path
                # Put all annotations in a dict according to the dataset name
                anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name, info=video_info)
                # Start processing annotations
                for idx, img in enumerate(
                        sorted(os.listdir(images_output_dir_path), key=lambda x: int(x.split(".")[0]))):
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": img,
                        "width": video_info["width"],
                        "height": video_info["height"],
                        "frame_id": int(img.split(".")[0]),
                        "video_id": video_name,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)

                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()
            else:
                print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")

    @staticmethod
    def ard_mav_rgb_anno_processor(anno:str, info: dict, mode: str) -> dict:
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        results = {}
        for frame_file in os.listdir(anno):
            a=DataReader.get_xml_file(os.path.join(anno,frame_file))
            frame=int(frame_file.split("_")[1].split(".")[0])
            detections = []
            # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
            bboxes = a.findall("object")
            for bbox in bboxes:
                xmin = bbox.find("bndbox").find("xmin").text
                ymin = bbox.find("bndbox").find("ymin").text
                xmax = bbox.find("bndbox").find("xmax").text
                ymax = bbox.find("bndbox").find("ymax").text
                d = (int(xmin), int(ymin), int(xmax) - int(xmin), int(ymax) - int(ymin))
                detections.append(
                    DatasetProcessor.output_form(d, info["width"], info["height"], mode)
                )
            results[frame] = detections
        return results

    @staticmethod
    def drone_dataset_uav_rgb_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT
        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT

        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)

        # For each vide
        suffix=""
        for media_dir in media_dir_list:
            if "xml" in media_dir:
                suffix=".xml"
            elif "txt" in media_dir:
                suffix=".txt"
            else:
                raise ValueError(f"Unknown dataset type: {dataset_name}")

            anno_dir = media_dir
            anno_direct_dir_path = os.path.join(dataset_object.anno_root_path, anno_dir,anno_dir)
            media_direct_dir_path = os.path.join(dataset_object.media_root_path, media_dir,media_dir)
            images_list, _ = DataReader.get_media_list(media_direct_dir_path)
            images_list = sorted(images_list, key=lambda x: int("".join(re.findall(r"\d",x))))


            # Start loop for each video
            img_id_count = 1
            anno_id_count = 1
            images_output_dir_path = os.path.join(
                dataset_object.images_output_root_path,
                media_dir
            )
            anno_output_dir_path = os.path.join(
                dataset_object.anno_output_root_path,
                media_dir)

            tools.create_dir_if_not_exists(images_output_dir_path)
            tools.create_dir_if_not_exists(anno_output_dir_path)
            # Start loop of each image

            for idx, image in enumerate(images_list):
                image_name, image_path, _, _,_, image_info = DatasetProcessor.out_put_initialization(
                    dataset_object, image,
                    image_single_dir=True,
                    media_direct_dir_path=media_direct_dir_path,
                    anno_direct_dir_path=anno_direct_dir_path)
                # Refresh the images if necessary
                if dataset_object.image_refresh_tag:
                    tools.image_copy(image_path, images_output_dir_path)
                else:
                    print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                # Refresh the annotations if necessary

                if dataset_object.anno_refresh_tag:
                    anno_path = os.path.join(anno_direct_dir_path, image_name + suffix)
                    if suffix==".xml":
                        anno = DataReader.get_xml_file(anno_path)
                    elif suffix==".txt":
                        anno = DataReader.get_txt_file(anno_path)
                    else:
                        raise TypeError(f"Unknown annotation type: {suffix}")

                    # Put all annotations in a dict according to the dataset name
                    anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                          info=image_info, idx=idx)
                    if len(anno_all_dict.keys()) == 0:
                        continue
                    # Start processing annotations
                    # Add image info to the annotations file.
                    image_dict = {
                        "id": int(img_id_count),
                        "file_path": image,
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "frame_id": idx+1,
                        "video_id": -1,
                        "data_type": dataset_object.dataset_type
                    }
                    dataset_object.coco_anno_dict["images"].append(image_dict)
                    img_id_count += 1

                    if image_dict["frame_id"] not in anno_all_dict.keys():
                        continue
                    for bboxes in anno_all_dict[image_dict["frame_id"]]:
                        anno_dict = {
                            "id": int(anno_id_count),
                            "image_id": image_dict["id"],
                            "category_id": 0,
                            "bbox": list(bboxes),
                            "iscrowd": 0
                        }
                        anno_id_count += 1
                        dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                else:
                    print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
            # Append category information
            dataset_object.coco_anno_dict["categories"].append({
                "id": 0,
                "name": "drone",
            })
            with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                json.dump(dataset_object.coco_anno_dict, f, indent=4)
            dataset_object.reset_coco()

    @staticmethod
    def drone_dataset_uav_rgb_anno_processor(anno, info: dict, mode: str,idx:int) -> dict:
        results = {}
        detections = []
        frame=idx+1
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        if isinstance(anno,list):
            for id, a in enumerate(sorted(anno)):
                rect=a.strip().split(" ")
                if "NaN" in rect:
                    continue
                else:
                    xmin=float(rect[1])
                    ymin=float(rect[2])
                    w=float(rect[3])
                    h=float(rect[4])

                detections.append(
                    DatasetProcessor.output_form((xmin,ymin,w,h), info["width"], info["height"], mode)
                )

        else:
            bboxes = anno.findall("object")
            for bbox in bboxes:
                xmin = bbox.find("bndbox").find("xmin").text
                ymin = bbox.find("bndbox").find("ymin").text
                xmax = bbox.find("bndbox").find("xmax").text
                ymax = bbox.find("bndbox").find("ymax").text
                d = (float(xmin), float(ymin), float(xmax) - float(xmin), float(ymax) - float(ymin))
                detections.append(
                    DatasetProcessor.output_form(d, info["width"], info["height"], mode)
                )
        results[frame] = detections
        return results

    @staticmethod
    def midgard_rgb_processor(dataset_object: Dataset, dataset_name: str, ffmpeg_dict=None, **kwargs):
        """
        Process purdue dataset to COCO format.
        Args:
            dataset_object: dataset object
            dataset_name: dataset name
            ffmpeg_dict: ffmpeg parameters
        Returns:
            No return value.
        """

        if ffmpeg_dict is None:
            ffmpeg_dict = DEFAULT_FFMPEG_DICT

        media_dir_list = DataReader.get_media_dir_list(dataset_object.media_root_path)

        # For each vide
        for media_class_dir in media_dir_list:
            media_class_dir_path = os.path.join(dataset_object.media_root_path, media_class_dir)
            for media_dir in DataReader.get_media_dir_list(media_class_dir_path):
                anno_dir = media_dir
                anno_direct_dir_path = os.path.join(dataset_object.anno_root_path, media_class_dir,anno_dir,"annotation")
                media_direct_dir_path = os.path.join(dataset_object.media_root_path, media_class_dir, media_dir,"images")
                images_list, _ = DataReader.get_media_list(media_direct_dir_path)
                images_list = sorted(images_list, key=lambda x: x.split(".")[0].split("_")[1])

                # Start loop for each video
                img_id_count = 1
                anno_id_count = 1
                images_output_dir_path = os.path.join(
                    dataset_object.images_output_root_path,
                    media_dir
                )
                anno_output_dir_path = os.path.join(
                    dataset_object.anno_output_root_path,
                    anno_dir)

                tools.create_dir_if_not_exists(images_output_dir_path)
                tools.create_dir_if_not_exists(anno_output_dir_path)
                # Start loop of each image

                for idx, image in enumerate(images_list):
                    image_name, image_path, _, _, _, image_info = DatasetProcessor.out_put_initialization(
                        dataset_object, image,
                        image_single_dir=True,
                        media_direct_dir_path=media_direct_dir_path,
                        anno_direct_dir_path=anno_direct_dir_path)
                    # Refresh the images if necessary
                    if dataset_object.image_refresh_tag:
                        tools.image_copy(image_path, images_output_dir_path)
                    else:
                        print(f"Image Refresh tag is False, {dataset_path} skip image processing.")

                    # Refresh the annotations if necessary

                    if dataset_object.anno_refresh_tag:
                        image_seq=image_name.split(".")[0].split("_")[1]
                        anno_path = os.path.join(anno_direct_dir_path, f"annot_{image_seq}"+dataset_object.anno_suffix)
                        anno=DataReader.get_csv_file(anno_path)
                        print(anno_path)

                        # Put all annotations in a dict according to the dataset name
                        anno_all_dict = DatasetProcessor.anno_processor_entry(anno, dataset_name=dataset_name,
                                                                              info=image_info, idx=idx)
                        if len(anno_all_dict.keys()) == 0:
                            continue
                        # Start processing annotations
                        # Add image info to the annotations file.
                        image_dict = {
                            "id": int(img_id_count),
                            "file_path": image,
                            "width": image_info["width"],
                            "height": image_info["height"],
                            "frame_id": int(image_name.split(".")[0].split("_")[1])+1,
                            "video_id": -1,
                            "data_type": dataset_object.dataset_type
                        }
                        dataset_object.coco_anno_dict["images"].append(image_dict)
                        img_id_count += 1

                        if image_dict["frame_id"] not in anno_all_dict.keys():
                            continue
                        for bboxes in anno_all_dict[image_dict["frame_id"]]:
                            anno_dict = {
                                "id": int(anno_id_count),
                                "image_id": image_dict["id"],
                                "category_id": 0,
                                "bbox": list(bboxes),
                                "iscrowd": 0
                            }
                            anno_id_count += 1
                            dataset_object.coco_anno_dict["annotations"].append(anno_dict)
                    else:
                        print(f"Annotation refresh tag is False, {dataset_path} skip anno processing.")
                # Append category information
                dataset_object.coco_anno_dict["categories"].append({
                    "id": 0,
                    "name": "drone",
                })
                with open(os.path.join(anno_output_dir_path, "annotations.json"), "w") as f:
                    json.dump(dataset_object.coco_anno_dict, f, indent=4)
                dataset_object.reset_coco()

    @staticmethod
    def midgard_rgb_anno_processor(anno:pandas.DataFrame, info: dict, mode: str, idx: int) -> dict:
        results = {}
        detections = []
        frame = idx + 1
        # Formed in: time_layer: 1798 detections: (y_min,x_min,y_max,x_max)
        for i in range(len(anno)):
            row=anno.iloc[i]
            xmin = int(row[1])
            ymin = int(row[2])
            w = int(row[3])
            h = int(row[4])
            detections.append(
                DatasetProcessor.output_form((xmin, ymin, w, h), info["width"], info["height"], mode)
            )

        results[frame] = detections
        return results


if __name__ == '__main__':
    start_time=time.time()
    parser=argparse.ArgumentParser(description="Merger inputs config")
    parser.add_argument("--dataset-name",default="purdue_rgb")
    args=parser.parse_args()

    test_dataset=args.dataset_name

    if test_dataset=="purdue_rgb":
        dataset_path="/home/king/PycharmProjects/DataMerger/Data/PURDUE_rgb"
        d=Dataset(dataset_path,
                  image_refresh_tag=True,
                  anno_refresh_tag=True,
                  media_dir_in_root=os.path.join("Videos","Videos"),
                  anno_dir_in_root=os.path.join("Video_Annotation","Video_Annotation"),
                  anno_suffix="_gt.txt")
        DatasetProcessor.purdue_rpg_processor(dataset_object=d,dataset_name=test_dataset)
    elif test_dataset=="real_world_rgb":
        dataset_path="/home/king/PycharmProjects/DataMerger/Data/Real Word Dataset_rgb"
        d=Dataset(dataset_path,
                  image_refresh_tag=True,
                  anno_refresh_tag=True,
                  media_dir_in_root=os.path.join("DroneTrainDataset","Drone_TrainSet"),
                  anno_dir_in_root=os.path.join("DroneTrainDataset","Drone_TrainSet_XMLs"),
                  anno_suffix=".xml")
        DatasetProcessor.real_world_rgb_processor(dataset_object=d,dataset_name=test_dataset)

    elif test_dataset=="anti_uav_rgbt_mix":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Anti-UAV-RGBT_mix"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=["test","train","val"],
                    anno_dir_in_root=["test","train","val"],
                    anno_suffix=".json")
        DatasetProcessor.anti_uav_rgbt_mix_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset=="anti_uav410_thermal":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Anti-UAV410_thermal"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=["test","train","val"],
                    anno_dir_in_root=["test","train","val"],
                    anno_suffix="IR_label.json")
        DatasetProcessor.anti_uav410_thermal_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset=="jet_fly_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Jet-Fly_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root="JPEGImages",
                    anno_dir_in_root="Annotations",
                    anno_suffix=".xml")
        DatasetProcessor.jet_fly_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset=="fdb_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Flight Dynamics-Based Recovery of a UAV Trajectory Using Ground Cameras_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=os.path.join("cvpr15","cvpr15","videos"),
                    anno_dir_in_root=os.path.join("cvpr15","cvpr15","annotations"),
                    anno_suffix=".txt")
        DatasetProcessor.fdb_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "youtube_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/youtube_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root="image&label",
                    anno_dir_in_root="image&label",
                    anno_suffix=".mat")
        DatasetProcessor.youtube_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "3rd_anti_uav_thermal":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/3rd_Anti-UAV_train_val_thermal"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=["train","validation"],
                    anno_dir_in_root="",
                    anno_suffix=".json")
        DatasetProcessor.third_anti_uav_thermal_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "uav123_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Dataset_UAV123_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=os.path.join("UAV123","data_seq","UAV123"),
                    anno_dir_in_root=os.path.join("UAV123","anno","UAV123"),
                    anno_suffix=".txt")
        DatasetProcessor.uav123_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "ard_mav_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/ARD-MAV_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root="videos",
                    anno_dir_in_root="Annotations",
                    anno_suffix=".xml")
        DatasetProcessor.ard_mav_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "drone_dataset_uav_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Drone-Dataset(UAV)_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root="archive",
                    anno_dir_in_root="archive",
                    anno_suffix="")
        DatasetProcessor.drone_dataset_uav_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    elif test_dataset == "midgard_rgb":
        dataset_path = "/home/king/PycharmProjects/DataMerger/Data/Midgard_rgb"
        d = Dataset(dataset_path,
                    image_refresh_tag=True,
                    anno_refresh_tag=True,
                    media_dir_in_root=os.path.join("MIDGARD","MIDGARD"),
                    anno_dir_in_root=os.path.join("MIDGARD","MIDGARD"),
                    anno_suffix=".csv")
        DatasetProcessor.midgard_rgb_processor(dataset_object=d, dataset_name=test_dataset)

    end_time=time.time()
    print(f"Total time is {end_time-start_time} s")