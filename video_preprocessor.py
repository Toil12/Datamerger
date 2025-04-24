from moviepy import VideoFileClip

def video_re(input_video_path, output_video_path, target_resolution):
    # 视频文件路径
    input_video_path = 'input.mp4'
    # 输出视频路径
    output_video_path = 'output.mp4'
    # 目标分辨率
    target_resolution = (960, 540)
    # 加载视频文件
    video = VideoFileClip(input_video_path)
    # 转换分辨率
    video = video.resize(target_resolution)
    # 保存输出视频
    video.write_videofile(output_video_path, codec='libx264')
    # 释放资源
    video.close()

if __name__ == '__main__':
    input_path="/home/king/PycharmProjects/detect_track/buffer/self/videos/MVI_0516.mp4"
    output_path="/home/king/PycharmProjects/detect_track/buffer/self/videos_processed/MVI_0516.mp4"
    target_resolution=(960,540)
    video_re(input_path,output_path,target_resolution)