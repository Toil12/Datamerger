import os
import scipy
import shutil
import argparse
import json
import xml.etree.ElementTree as ET

SUPPORT_IMAGE_FORMAT = (".jpg", ".jpeg", ".png", ".bmp")
SUPPORT_VIDEO_FORMAT = (".mp4", ".avi", ".mkv", ".mov",".avi")
TYPE_SUFFIX = ("_rgb", "_thermal", "_mix")

class DataReader:
    """
    Get video list and all images of each video.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_media_dir_list(root_dir_path:str)->[]:
        """
        Get all video directories in a root directory.
        Args:
            root_dir_path: root directory path
        Returns:
            video_dir_list: list of video directories
        """
        media_dir_list = []
        for d in os.listdir(root_dir_path): # list all directories in root_dir
            if not os.path.isdir(os.path.join(root_dir_path, d)):
                continue
            else:
                media_dir_list.append(d)
        print(f"Get media directories {len(media_dir_list)}, start processing")
        return media_dir_list

    @staticmethod
    def get_media_list(dir_path:str):
        """
        Get all images and videos in a directory.
        Args:
            dir_path: directory path
        Returns:
            img_list: list of images
            video_list: list of videos
        """
        img_list = []
        video_list = []
        for f in os.listdir(dir_path): # list all images in video_dir
            if os.path.isfile(os.path.join(dir_path, f)):
                if f.endswith(SUPPORT_IMAGE_FORMAT):
                    img_list.append(f)
                elif f.endswith(SUPPORT_VIDEO_FORMAT):
                    video_list.append(f)
            else:
                pass
        print(f"Get images {len(img_list)}, videos {len(video_list)}, start processing")
        return img_list, video_list

    @staticmethod
    def get_json_anno(json_path: str):
        """
        Read a JSON file and return its content.
        Args:
            json_path (str): The path to the JSON file.
        Returns:
            dict: A dictionary containing the parsed JSON data.
                  Returns None if the file is not found or an error occurs.
        """
        try:
            with open(json_path, "r") as js_file:
                data = json.load(js_file)
            return data
        except FileNotFoundError:
            print(f"The file {json_path} was not found.")
            return None
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from {json_path}.")
            return None

    @staticmethod
    def get_mat_file(mat_path: str):
        """
        Read a .mat file and return its content.

        Args:
            mat_path (str): The path to the .mat file.

        Returns:
            dict: A dictionary containing the variables stored in the .mat file.
                  Returns None if the file is not found or an error occurs.
        """
        try:
            from scipy.io import loadmat
            data = loadmat(mat_path)
            return data
        except FileNotFoundError:
            print(f"The file {mat_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading {mat_path}: {e}")
            return None

    @staticmethod
    def get_xml_file(xml_path: str):
        """
        Read an XML file and return the root element.

        Args:
            xml_path (str): The path to the XML file.

        Returns:
            xml.etree.ElementTree.Element: The root element of the XML tree.
            None: If the file is not found or an error occurs.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return root
        except FileNotFoundError:
            print(f"The file {xml_path} was not found.")
            return None
        except ET.ParseError:
            print(f"Failed to parse XML from {xml_path}.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading {xml_path}: {e}")
            return None

    @staticmethod
    def get_txt_file(txt_path: str):
        """
        Read a text file and return its content as a string.

        Args:
            txt_path (str): The path to the text file.

        Returns:
            str: The content of the text file.
            None: If the file is not found or an error occurs.
        """
        try:
            with open(txt_path, "r", encoding="utf-8") as txt_file:
                content = txt_file.readlines()
            return content
        except FileNotFoundError:
            print(f"The file {txt_path} was not found.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading {txt_path}: {e}")
            return None

    @staticmethod
    def support_video_format()->tuple:
        return SUPPORT_VIDEO_FORMAT

    @staticmethod
    def support_image_format()->tuple:
        return SUPPORT_IMAGE_FORMAT

if __name__ == '__main__':
    root_pth="Data"
    test_file_path="/home/king/PycharmProjects/DataMerger/Data/Jet-Fly_rgb/Annotations/010/0100002.xml"
    reader=DataReader(root_pth)

    # print(reader.get_xml_file(test_file_path).find("object").find("bndbox").find("xmin").text)