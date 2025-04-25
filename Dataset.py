class Dataset:
    def __init__(self,data_dir):
        self.data_dir=data_dir
        self.dataset_dict={}
        self.dataset_info_initialize()