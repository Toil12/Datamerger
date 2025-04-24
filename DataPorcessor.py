import torch
import shutil
import os
from DataReader import DataReader

DEFAULT_DATASET_ROOT="./Data"

class DataProcessor:
    def __init__(self,root_dir=DEFAULT_DATASET_ROOT):
        self.root_dir=root_dir
        self.dataset_list=[]

        #
        self.fetch_data_info()
        self.build_annotation()

    def fetch_data_info(self,update_tag=False):
        """
        Read dataset name in dataset_list.txt or create dataset_list.txt.
        Args:
            update_tag: if True, update dataset_list.txt.
        Returns:
            No return value.
        """
        # empty_file_tag=os.path.getsize(os.path.join(self.root_dir,"dataset_list.txt"))

        if update_tag:
            for dataset in os.listdir(self.root_dir):
                if os.path.isdir(os.path.join(self.root_dir,dataset)):
                    self.dataset_list.append(dataset)
            try:
                with open(os.path.join(self.root_dir,"dataset_list.txt"), 'w') as file:
                    # Opening in 'w' mode truncates the file, effectively clearing its content
                    for item in self.dataset_list:
                        file.write(item.strip() + '\n')
            except Exception as e:
                print(f"{e}")
        else:
            try:
                with open(os.path.join(self.root_dir,"dataset_list.txt"),"r") as f:
                    for line in f.readlines():
                        self.dataset_list.append(line.strip())
            except Exception as e:
                print(f"When reading existing file, meet error {e}")

    # def annotation_structure_initialize(self,update_tag=False):
    #     """
    #     Create annotation info directory for each dataset.
    #     Args:
    #         update_tag: if True, update annotation info directory.
    #     Returns:
    #         No return value.
    #     """
    #     for dataset in self.dataset_list:
    #         dataset_path=os.path.join(self.root_dir,dataset)
    #         try:
    #             if update_tag:
    #                 shutil.rmtree(os.path.join(dataset_path,self.anno_info_dir))
    #             if self.anno_info_dir not in os.listdir(dataset_path):
    #                 os.makedirs(os.path.join(dataset_path,self.anno_info_dir))
    #         except Exception as e:
    #             print(f"An error occurred while processing {e}")

    def build_annotation(self):
        for dataset in self.dataset_list:
            dataset_path=os.path.join(self.root_dir,dataset)



if __name__ == '__main__':
    processor=DataProcessor()
    print(processor.dataset_list)