import copy
import json
import time
import os
import tools


DEFAULT_DATASET_ROOT="./Data"

class DataProcessor:
    def __init__(self,root_dir=DEFAULT_DATASET_ROOT,**kwargs):
        self.root_dir=root_dir
        self.dataset_list=[]
        self.dataset_list_path = os.path.join(self.root_dir, "dataset_list.txt")
        self.results_dir=os.path.join(self.root_dir,"all_results")
        self.results_summary_json = {"info":{},
                                     "licenses":[],
                                     "images": [],
                                     "annotations": [],
                                     "categories": []
                                    }
        #
        tools.create_dir_if_not_exists(self.results_dir)

    def fetch_all_data_info(self, update_tag=False):
        """
        Read dataset name in dataset_list.txt or create dataset_list.txt.
        Args:
            update_tag: if True, update dataset_list.txt.
        Returns:
            No return value.
        """

        if update_tag:
            for dataset in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir,dataset)):
                    self.dataset_list.append(dataset)
            try:
                with open(self.dataset_list_path, 'w') as file:
                    # Opening in 'w' mode truncates the file, effectively clearing its content
                    for item in self.dataset_list:
                        file.write(item.strip() + '\n')
            except Exception as e:
                print(f"{e}")
        else:
            try:
                with open(self.dataset_list_path,"r") as f:
                    for line in f.readlines():
                        self.dataset_list.append(line.strip())
            except Exception as e:
                print(f"When reading existing file, meet error {e}")

    # def build_annotation(self):
    #     for dataset in self.dataset_list:
    #         dataset_path=os.path.join(self.root_dir,dataset)


    def all_dataset_summarization(self):
        start_time=time.time()
        summary_file_count=len(os.listdir(self.results_dir))

        #
        results_json_file_name=f"dataset_summary_{summary_file_count}.json"
        results_json_file_path=os.path.join(self.results_dir,results_json_file_name)

        img_id=1
        anno_id=1
        category_tag=False
        key_mapping={"file_path":"file_name"}

        for dataset in ["PURDUE_rgb","Jet-Fly_rgb"]:
            d_start_time=time.time()
            dataset_path=os.path.join(self.root_dir,dataset)
            anno_root_dir_path=os.path.join(dataset_path,"annotations_coco")
            media_root_dir_path=os.path.join(dataset_path,"images_coco")
            anno_dir_list=os.listdir(anno_root_dir_path)

            # Start from the annotations file
            for anno_dir in anno_dir_list:
                anno_path=os.path.join(anno_root_dir_path,anno_dir,"annotations.json")
                with open(anno_path) as f:
                    anno=json.load(f)
                # Add categories once
                if not category_tag:
                    self.results_summary_json["categories"]=anno["categories"]
                    category_tag=True
                # Image 'file_path' to 'file_name'
                for img in anno["images"]:
                    img = {key_mapping.get(k, k): v
                            for k, v in img.items()
                            }
                    img["file_name"]=os.path.join(media_root_dir_path,anno_dir,img["file_name"])
                    if not os.path.isfile(img["file_name"]):
                        k=img["file_name"]
                        raise TypeError(f"{k} is wrong path")
                    img["video_id"]=os.path.join(dataset,anno_dir)
                    img["related_anno"] =[]
                    # To find and set the new image id
                    old_img_id=copy.deepcopy(img["id"])
                    new_img_id=img_id
                    img["id"]=new_img_id
                    #
                    searched_tag=False
                    for anno_c in anno["annotations"]:
                        if anno_c["image_id"]==old_img_id:
                            anno_c["image_id"]=new_img_id
                            anno_c["id"]=anno_id
                            anno_c["bbox"]=list(map(int,anno_c["bbox"]))
                            if len(anno_c["bbox"])==0:
                                raise ValueError(img["file_name"]," should not be empty")
                            if any(c>max(img["width"],img["height"]) for c in anno_c["bbox"]):
                                position=anno_c["bbox"]
                                raise ValueError(f"{position} with oversized position")
                            # Delete the empty annotations
                            # TODO cheche this filter
                            if anno_c["bbox"]==[0,0,0,0] or anno_c["bbox"]==[]:
                                continue
                            img["related_anno"].append(anno_id)
                            self.results_summary_json["annotations"].append(anno_c)
                            anno_id+=1
                            searched_tag=True
                        elif searched_tag:
                            break
                        else:
                            continue
                    # If not found, skip this image
                    if not searched_tag:
                        continue
                    else:
                        self.results_summary_json["images"].append(img)
                        img_id += 1
            d_end_time=time.time()
            print(f"Dataset {dataset} is finished in {d_end_time-d_start_time} s")
        with open(results_json_file_path,"w",encoding="utf-8") as f:
            json.dump(self.results_summary_json,f,ensure_ascii=False,indent=4)
        end_time=time.time()
        print(f"Get final annotations in {results_json_file_path} with {end_time-start_time} s")


if __name__ == '__main__':
    processor=DataProcessor()
    processor.fetch_all_data_info()
    processor.all_dataset_summarization()