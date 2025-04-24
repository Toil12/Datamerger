import os
import torch
import cv2
import numpy as np
from torchvision import transforms

class LocalCOCO:
    def __init__(self,data_dir:str,mode:str):
        self.data_dir=os.path.join(data_dir,mode)
        self.coco_dict = {"images": [],
                          "annotations": [],
                          "categorise": []
                          }
        self.image_info_initialize()
        self.annotation_info_initialize()
    def image_info_initialize(self):
        file_count = 1
        for video_dir in os.listdir(self.data_dir):  # video dir is like "0"
            video_path = os.path.join(self.data_dir, video_dir)  # video path is like 0.jpg
            # print(os.listdir(video_path))
            for idx, image_name in enumerate(sorted(os.listdir(video_path),key=lambda x:int(x.split(".")[0]))):
                image_path = os.path.join(video_path, image_name)
                file_name = os.path.join(video_dir, image_name)
                image_dict = {"file_name": file_name,
                              "id": file_count,
                              "video_id": int(video_dir),
                              "frame_id": idx + 1}
                self.coco_dict["images"].append(image_dict)
                file_count += 1

    def annotation_info_initialize(self):
        pass

    def get_id_number(self):
        ids=[]
        item_list=self.coco_dict["images"]
        for item in item_list:
            ids.append(item["id"])
        return ids

    def get_from_index(self,index):
        info=[]
        transform=transforms.ToTensor()
        for item in self.coco_dict["images"]:
            if item["id"]==index+1:
                image_path = item["file_name"]
                img_source=cv2.imread(os.path.join(self.data_dir,image_path))
                img_source=torch.unsqueeze(transform(img_source),dim=0)
                img_source.permute(0,2,3,1)

                info.append(torch.tensor(img_source.shape[2]))
                info.append(torch.tensor(img_source.shape[3]))
                info.append(torch.tensor(item["frame_id"]))
                info.append(str(item["video_id"]))
                info.append(str(item["file_name"]))
                return img_source,info,image_path
            else:
                raise IndexError(f"The given index {index} is out of the video frames")