"""
coding:utf-8
@Time       :2024/9/14 9:31
@Author     :ywLi
@Institute  :DonghaiLab
"""
import shutil
import platform
import time

import cv2
import os

import xml.dom.minidom as xmldom
import os.path as osp

from sklearn.model_selection import train_test_split
from Image_preprocessor import ImagePreProcessor as imgpp
from tools import clean_create_dir_files



system=platform.system().lower()
slash=""
if system == 'windows':
    slash="\\"
elif system == 'linux':
    slash="/"

VIDEOS_FOLDER_NAME="self"
VIDEOS_ROOT = os.path.join(os.getcwd(), "datasets_original", VIDEOS_FOLDER_NAME)
ANNO_PATH = os.path.join(os.getcwd(), "datasets_original", VIDEOS_FOLDER_NAME)
IMAGES_ROOT = os.path.join(os.getcwd(), "datasets_local", VIDEOS_FOLDER_NAME, "video_images")
OUTPUT_IMAGES_DIR = "all_images"
OUTPUT_ANNOTATION_DIR = "all_annotations"
BASE_PATH = os.curdir
import time
import ffmpeg

def video2imgs_ffmpeg(video_path,
                      image_root,
                      file_name,
                      frames_num=1,
                      filter_scale=(960,540),
                      frame_gap=1
                      ):

    # file_name=image_path.split(slash)[-1].split(".")[0]
    image_path=os.path.join(image_root,file_name)
    clean_create_dir_files(image_path)
    image_path_template = os.path.join(image_path, f"%d.jpg")

    video_root=f"{slash}".join(video_path.split(slash)[:-2])

    video_output_path=os.path.join(video_root,f"videos_processed",file_name+"_r"+".mp4")
    if frames_num<=5:
        print("v root is", video_root)
        print("v output path is",video_output_path)
    # Extract frames from the video and save them as images according to gaps ad rescale the video
    if frames_num==-1:
        (
            ffmpeg
            .input(video_path)
            .filter('framestep', frame_gap)
            .filter("scale", filter_scale[0], filter_scale[1])
            .output(image_path_template,crf=0, preset='slow')
            .overwrite_output()
            .run()
        )
    else:
        (
            ffmpeg
            .input(video_path)
            .filter('frame_step',5)
            .filter("scale", filter_scale[0], filter_scale[1])
            .output(image_path_template, vframes=frames_num,crf=0, preset='slow')
            .overwrite_output()
            .run()
        )
    (
        ffmpeg
        .input(video_path)
        .filter("scale", filter_scale[0], filter_scale[1])
        .output(video_output_path,b='0')
        .overwrite_output()
        .run()
    )
    name_modify_by_frame_gaps(image_path,frame_gap)

    return video_output_path

def video2imgs(video_path, img_root,image_dir_name, count_limit=-1,frame_gaps=1):
    # Clear directory and files if exists
    img_path=os.path.join(img_root,image_dir_name)
    clean_create_dir_files(img_path)
    video_dir=video_path.split(".")[0].split(slash)[-1]
    cap = cv2.VideoCapture(video_path)    # 获取视频
    judge = cap.isOpened()                 # 判断是否能打开成功
    print(judge)
    fps = cap.get(cv2.CAP_PROP_FPS)      # 帧率，视频每秒展示多少张图片
    print('fps:',fps)

    frames = 1                           # 用于统计所有帧数
    count = 1                            # 用于统计保存的图片数量

    while judge and (count_limit == -1 or count <= count_limit):
        flag, frame = cap.read()         # 读取每一张图片 flag表示是否读取成功，frame是图片
        if not flag:
            print(flag)
            print(f"Process {video_dir} finished!")
            break
        else:
            if frames % frame_gaps == 0:         # 每隔1帧抽一张
                imgname =  str(count).rjust(4,'0') + ".jpg"
                newPath = os.path.join(img_path, imgname)
                # print(imgname,newPath)
                h,w,c=frame.shape
                resized_frame=cv2.resize(frame,(w//4,h//4))
                cv2.imwrite(newPath, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # cv2.imencode('.jpg', frame)[1].tofile(newPath)
                count += 1
        frames += 1
    cap.release()
    print("共有 %d 张图片"%(count-1))

def name_modify_by_frame_gaps(frames_path, gap):
    '''
    Modify the name of the frames by the frame gaps
    :param frames_path:
    :param gap:
    :return:
    '''
    for old_filename in os.listdir(frames_path):
        file_path = os.path.join(frames_path, old_filename)
        if os.path.isfile(file_path):
            new_filename = "{:05d}".format((int(old_filename.split(".")[0])-1)*gap+1)+"."+old_filename.split(".")[1]
            new_file_path = os.path.join(frames_path, new_filename)
            os.rename(file_path, new_file_path)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def annotation_xml_to_yolo(images_root,anno_root)->None:
    """
    The images will be put under path/video_images/directory_under_video_name
    :param images_root: The root of the dataset from video transformed images.
    :param anno_root: The root of annotation.
    """

    # delete the buffer directions
    shutil.rmtree(os.path.join(BASE_PATH,OUTPUT_IMAGES_DIR))
    shutil.rmtree(os.path.join(BASE_PATH,OUTPUT_ANNOTATION_DIR))
    os.mkdir(os.path.join(BASE_PATH,OUTPUT_IMAGES_DIR))
    os.mkdir(os.path.join(BASE_PATH,OUTPUT_ANNOTATION_DIR))
    # start re-write
    for video_dir in os.listdir(anno_root):
        for file in os.listdir(os.path.join(anno_root,video_dir)):
            xml_file_path=os.path.join(anno_root,video_dir,file)
            # read xml files
            xml_file = xmldom.parse(xml_file_path)

            # print(f)
            eles = xml_file.documentElement
            # try, if no object, continue
            try:
                xmin = float(eles.getElementsByTagName("xmin")[0].firstChild.data)
                xmax = float(eles.getElementsByTagName("xmax")[0].firstChild.data)
                ymin = float(eles.getElementsByTagName("ymin")[0].firstChild.data)
                ymax = float(eles.getElementsByTagName("ymax")[0].firstChild.data)
            except Exception as e:
                print(f"{file} error")
                continue

            width=float(eles.getElementsByTagName("width")[0].firstChild.data)
            height = float(eles.getElementsByTagName("height")[0].firstChild.data)

            # transform to yolo form
            yolo_x = (xmin +  xmax) / (2*width)
            yolo_y = (ymin + ymax) / (2*height)
            yolo_w = (xmax-xmin) / width
            yolo_h = (ymax-ymin) / height


            with open(f'{BASE_PATH}/{OUTPUT_ANNOTATION_DIR}/{file.split(".")[0]}.txt', mode='w') as f:
                f.write("0" + ' ')
                f.write(str(yolo_x) + ' ')
                f.write(str(yolo_y) + ' ')
                f.write(str(yolo_w) + ' ')
                f.write(str(yolo_h))

            # print(type(images_root),type(video_dir),type(file))
            image_path=os.path.join(images_root,video_dir,file.split(".")[0]+".jpg")

            shutil.copy(image_path,OUTPUT_IMAGES_DIR)

def data_split(agg_pars=None):
    if agg_pars is None:
        agg_pars = {
            "gray": 0,
            "hist": 0,
            "lap": 0
        }
    new_data_images_path=os.path.join(os.curdir,"datasets_processed","ARD-MAV","images")
    new_data_anno_path=os.path.join(os.curdir,"datasets_processed","ARD-MAV","labels")

    output_images_dir=OUTPUT_IMAGES_DIR
    output_anno_dir=OUTPUT_ANNOTATION_DIR



    shutil.rmtree(os.path.join(new_data_images_path,"train"))
    shutil.rmtree(os.path.join(new_data_images_path, "val"))
    shutil.rmtree(os.path.join(new_data_anno_path, "train"))
    shutil.rmtree(os.path.join(new_data_anno_path, "val"))

    os.mkdir(os.path.join(new_data_images_path,"train"))
    os.mkdir(os.path.join(new_data_images_path,"val"))
    os.mkdir(os.path.join(new_data_anno_path, "train"))
    os.mkdir(os.path.join(new_data_anno_path, "val"))

    # Split the data into training and test
    annotations=os.listdir(output_anno_dir)
    images=os.listdir(output_images_dir)
    image_annotation_tuples=list(zip(images,annotations))
    train_tuples, val_tuples = train_test_split(image_annotation_tuples,
                                                 train_size=0.8,
                                                 test_size=0.2,
                                                 shuffle=False
                                                 )

    # Make annotations and images as pairs in training set
    for t in train_tuples:
        with open(osp.join(output_anno_dir, f"{t[1]}")) as f:
            # Drop data which is not with a target in the view, pos in positions < 0
            drop_tag=False
            positions=f.read().split(" ")
            for pos in positions[1:]:
                pos=float(pos)
                if pos<0:
                    drop_tag=True
                    break
        if drop_tag:
            continue
        else:
            img_path=os.path.join(output_images_dir, t[0])

            image=cv2.imread(img_path)

            image=imgpp.main_process(image,agg_pars)
            #
            image_id,file_id=img_path.split(slash)[-1:-3:-1]
            cv2.imwrite(osp.join(new_data_images_path,"train",f"{image_id.split('.')[0]}.jpg"),image)
            shutil.copy(osp.join(output_anno_dir, f"{t[1]}"), osp.join(new_data_anno_path, "train", f"{t[1]}"))

    # Make annotations and images as pairs in validation set
    for t in val_tuples:
        with open(os.path.join(output_anno_dir, f"{t[1]}")) as f:
            # Drop data which is not with a target in the view, pos in positions < 0
            drop_tag = False
            positions = f.read().split(" ")
            for pos in positions[1:]:
                pos = float(pos)
                if pos < 0:
                    drop_tag = True
                    break
        if drop_tag:
            continue
        else:
            img_path =os.path.join(output_images_dir, t[0])
            image = cv2.imread(img_path)
            image_id, file_id = img_path.split(slash)[-1:-3:-1]
            cv2.imwrite(osp.join(new_data_images_path, "val", f"{image_id.split('.')[0]}.jpg"), image)
            shutil.copy(osp.join(output_anno_dir, f"{t[1]}"), osp.join(new_data_anno_path, "val", f"{t[1]}"))

def run(video_root=VIDEOS_ROOT,
        images_root=IMAGES_ROOT,
        count_limit=-1,
        frame_intervals=1,
        viedeo_reshape=(960,540)):

    for dir_name in os.listdir(video_root):
        video_path=os.path.join(video_root,dir_name)
        img_dir_name=dir_name.split(".")[0]
        img_dir_path=os.path.join(images_root,img_dir_name)
        video2imgs(video_path,img_dir_path,count_limit,frame_intervals)

def run_ffmpeg(video_root=VIDEOS_ROOT,
            images_root=IMAGES_ROOT,
            frames_num=1,
            filter_scale=(960,540),
            frame_gaps=1):
    """
    :param video_root: the video root path
    :param images_root: the images root path
    :param frames_num: the frames number to be extracted from the video
    :param frame_gaps:
    :param filter_sacle:
    :return:
    """
    p_video_path=""
    print("Read original video from: ",video_root)
    if os.path.isfile(video_root):
        dir_name=video_root.split(slash)[-1].split(".")[0]
        video_path = video_root
        # Get the folder name from video
        img_dir_name = dir_name.split(".")[0]
        # img_dir_path = os.path.join(images_root, img_dir_name)
        p_video_path=video2imgs_ffmpeg(video_path,
                                       images_root,
                                       img_dir_name,
                                       frames_num,
                                       filter_scale,
                                       frame_gaps)

    elif os.path.isdir(video_root):
        for dir_name in os.listdir(video_root):
            video_path = os.path.join(video_root, dir_name)
            img_dir_name = dir_name.split(".")[0]
            img_dir_path = os.path.join(images_root, img_dir_name)
            p_video_path=video2imgs_ffmpeg(video_path, images_root, img_dir_name,frames_num)

    else:
        raise Exception("Invalid video_root format.")
    return p_video_path

if __name__ == '__main__':

    print("start process")
    start_time=time.time()
    # Videos to frames
    run_ffmpeg(video_root="/home/king/PycharmProjects/detect_track/buffer/self/videos/MVI_0520_118_125.mp4",
               images_root="/home/king/PycharmProjects/detect_track/buffer/self/video_images",
               frames_num=-1,
               frame_gaps=5)
    end_time=time.time()
    print(f"spend {end_time-start_time}s")
