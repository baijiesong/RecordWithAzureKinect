# RecordWithAzureKinect
## This Project is used to record the video with audio by AzureKinects and PC 
### 1. Experimental platform
* Operating system: Windows 11
* Depth sensor: Azure kinect
* python version: 3.8
* opencv-python version: 4.8.1
* open3d version: 0.17
### 2. Install the sensor driver on Windows 11
* Folow this https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md
* If you want to test if it is installed, you can take **k4arecorder.exe** under sensor files.
### 3. Configure the project environment with conda
* Clone the project
  * `git clone https://github.com/baijiesong/RecordWithAzureKinect.git `
* Create python env
  * `conda create -n 3dRec python=3.8`
  * `conda activate 3dRec`
  * `pip install open3d`
  * `pip install opencv-python`
### 4. Usage
* You can have video with audio after launch
  * `python rgbd_recorder.py`
## Based on https://github.com/wangyouOVO/ReConWithAzureKinect
## 本项目使用深度相机用于采集带有深度信息的视频，同时录制声音
