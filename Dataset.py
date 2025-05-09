import os
import tools
class Dataset:
    def __init__(self,dataset_path:str,**kwargs):
        self.dataset_path=dataset_path
        self.new_anno_dir = "annotations_coco"
        self.new_img_dir = "images_coco"
        self.coco_anno_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.images_output_root_path:str=""
        self.anno_output_root_path:str=""
        self.dataset_type:str=""

        self.image_refresh_tag=kwargs.get("image_refresh_tag",False)
        self.anno_refresh_tag=kwargs.get("anno_refresh_tag",False)
        self.dataset_name=kwargs.get("dataset_name","")

        # For video or images data
        self.media_dir_in_root=kwargs.get("media_dir_in_root","videos")
        self.anno_dir_in_root=kwargs.get("anno_dir_in_root","annotations")
        self.anno_fixed_name=kwargs.get("anno_fixed_name","anno.txt")
        self.anno_suffix=kwargs.get("anno_suffix",".txt")

        self.extra_info=kwargs

        if isinstance(self.media_dir_in_root,list):
        #TODO get the correct media output
            self.media_root_path=[os.path.join(dataset_path,media_dir) for media_dir in self.media_dir_in_root]
            self.anno_root_path=[os.path.join(dataset_path,anno_dir) for anno_dir in self.anno_dir_in_root]
        else:
            self.media_root_path=os.path.join(dataset_path,self.media_dir_in_root)
            self.anno_root_path=os.path.join(dataset_path,self.anno_dir_in_root)

        self.initialization()


    def initialization(self)->None:
        """
        Initialization of dataset.
        """
        self.dataset_type = self.dataset_path.split("_")[-1]
        self.images_output_root_path = os.path.join(self.dataset_path, self.new_img_dir)
        self.anno_output_root_path = os.path.join(self.dataset_path, self.new_anno_dir)

        if self.image_refresh_tag:
            tools.clean_create_dir_files(self.images_output_root_path)
        else:
            tools.create_dir_if_not_exists(self.images_output_root_path)

        if self.anno_refresh_tag:
            tools.clean_create_dir_files(self.anno_output_root_path)
        else:
            tools.create_dir_if_not_exists(self.anno_output_root_path)

    def reset_coco(self):
        self.coco_anno_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }