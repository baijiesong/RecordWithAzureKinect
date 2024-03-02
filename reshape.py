
# ---------------------------------------------------------
#Author: Bai Jiesong , Shanghai University, 2023
# ---------------------------------------------------------
import subprocess
import os
import uuid
from PIL import Image

ffmpeg_path="F:\\ffmpeg-6.1-full_build\\ffmpeg-6.1-full_build\\bin\\ffmpeg.exe"
def resolution_picture(size):
    files = os.listdir('data/bai/raw/color/')
    out_num = len(files)
    for i in range(0,out_num):
        fileName = 'data/bai/raw/color/'+'{0:05d}.jpg'.format(i)    #循环读取所有的图片,假设以数字顺序命名
        image = Image.open(fileName)
        resized_image = image.resize(size,Image.ANTIALIAS)
        resized_image.save(fileName.split('.')[0] + '_1.' + fileName.split('.')[-1])

def crop_images(width, height):
    files = os.listdir('data/bai/raw/color/')
    
    out_num = len(files)     
    for i in range(0,out_num):
        fileName = 'data/bai/raw/depth/'+'{0:05d}.png'.format(i)    #循环读取所有的图片,假设以数字顺序命名
        dest_file='depth_maps/'+'{0}.png'.format(i)
        image =fileName
        crop_cmd = f"{ffmpeg_path} -i {image} -vf crop={width}:{height} {dest_file} -y"
        subprocess.run(crop_cmd, shell=True)   


def resolution_video(video_name: str, output_dir: str, size:tuple, bit_rate=2000):
    ext = os.path.basename(video_name).strip().split('.')[-1]
    # if ext not in ['mp4']:
    #     raise Exception('format error')
    newVideoName = os.path.join(
        output_dir, '{}.{}'.format(
            uuid.uuid1().hex, ext))
    #ff = ffmpeg(inputs={'{}'.format(video_name): None}, outputs={
    #    newVideoName: '-s {}*{} -b {}k'.format(size[0],size[1], bit_rate)})
   # ff.run()
    return newVideoName

if __name__ == "__main__":
    crop_images(720,720)