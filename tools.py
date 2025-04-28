import os
import shutil
import torch
import cv2
import ffmpeg

from DataReader import DataReader


def list_and_sort_files(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    return files

def create_dir_if_not_exists(root_path,name:str=""):
    """
    create a directory if not exists.
    if name is not empty, create a directory with name in root_path.
    if name is empty, create a directory in root_path.
    if directory already exists, print a message.
    if root_path is not a directory, raise a TypeError.
    Args:
        root_path: root path
        name: name of directory
    Returns:
        NO return value.
    """
    if name!="":
        dir_path=os.path.join(root_path,name)
    else:
        dir_path=root_path

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"Directory in path {dir_path} created.")
    else:
        print(f"Directory in path {dir_path} already exists.")


def clean_create_dir_files(folder_path):
    """
    clear all files in a folder if exists, otherwise create a new folder.
    if folder_path is not a directory, raise a TypeError.
    Args:
        folder_path: folder path
    Returns:
        No return value.
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Directory in path {folder_path} created.")
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f"Directory in path {folder_path} re-built.")

def print_gpu_memory_load():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
    else:
        print("No GPU available.")

def video2images(video_path, frames_path):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    else:
        clean_create_dir_files(frames_path)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{frames_path}/{i}.jpg", frame)
            i += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return i

def video2images_ffmpeg(video_path:str,output_path:str,**ffmpeg_args):
    image_path_template = os.path.join(output_path, f"%d.jpg")
    stream=ffmpeg.input(video_path)
    stream=stream.filter('framestep',ffmpeg_args["framestep"])
    stream=stream.output(image_path_template,crf=0, preset='slow')
    stream.run()

def get_video_info(video_path:str):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info={
        "frame_count":frame_count,
        "fps":fps,
        "width":width,
        "height":height
    }
    return info

def get_image_info(image_path:str):
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    info={
        "width":int(width),
        "height":int(height),
        "channels":int(channels)
    }
    return info

def image_copy(original_path:str,output_path:str):
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    # else:
    #     clean_create_dir_files(output_path)
    if original_path.endswith(DataReader.support_image_format()):
        shutil.copy(original_path,output_path)

if __name__ == '__main__':
    # v_path="/home/king/PycharmProjects/detect_track/buffer/self/videos/bird_drone.mp4"
    # o_path="/home/king/PycharmProjects/detect_track/buffer/self/video_images/bird_drone/test"
    # video_2_frames(v_path,o_path)

    ffmpeg_dict={
        "framestep": 1,
        "resize_w": 640,
        "resize_h": 640
    }
    output_path="results"
    video_path="bee.mp4"
    video2images_ffmpeg(video_path,output_path,**ffmpeg_dict)