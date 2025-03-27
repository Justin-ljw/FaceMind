# FaceMind
FaceMind：基于OpenCV + RetinaFace + ArcFace的实时人脸识别系统

## 技术和库

OpenCV：用于摄像头实时拍摄和图像处理。

InsightFace：用于人脸检测、对齐和识别。（RetinaFace + ArcFace）

SQLite：用于存储人脸数据和用户信息的数据库。

Gradio：用于创建前端界面。

## 工作流程

### 1.摄像头实时拍摄：
使用OpenCV调用本地摄像头并捕捉实时视频流。

### 2.人脸检测、对齐和识别
人脸识别的操作包括：

人脸检测（即从原始图像中检测出人脸目标框） -> 关键点提取（以关键点标识人脸的五官） -> 仿射变换对齐（通过平移、放缩、旋转等仿射变换把人脸图像转换成标准形式，以处理歪头等情况） -> 人脸识别（对人脸图像进行Embedding，如果与数据库已知人脸计算显示度来匹配）

这里使用InsightFace库来实现对实时视频流的人脸识别。InsightFace集成了RetinaFace和ArcFace，RetinaFace用于实现人脸检测、关键点提取和仿射变换对齐，ArcFace对人脸图像进行embedding。先用ratinaFace对摄像头拍到的图像实时进行人脸检测、提取关键点，并进行仿射变换把图像转换成标准形式，然后把图像给到ArcFace进行embedding，并用当前这个人脸embedding和数据库中所有的已知人脸embedding计算cos相似度，匹配最相似的已录入人脸（未见过的人脸输出“Unknown”），最后把目标框和人名输出回原始图像。

### 3.数据库支持：

使用SQLite数据库存储已知人脸的编码和用户信息。

支持摄像头实时拍摄和上传照片两种方式动态录入新的人脸，将新的人脸图片、编码和用户姓名存储到数据库中。

### 4.前端界面：

使用Gradio创建一个简单的Web应用，提供用户界面。

前端界面允许用户查看实时视频流、录入新的人脸、查看识别结果等。

## 快速开始
克隆仓库

git clone https://github.com/Justin-ljw/FaceMind.git

安装依赖

pip install -r requirements.txt

## Reference
RetinaFace 论文地址：https://arxiv.org/abs/1905.00641

ArcFace 论文地址：https://arxiv.org/abs/1801.07698

所用的ArcFace模型文件 FaceMind/.insightface/models/arcface_resnet100_wf4m.pt 来自 https://huggingface.co/minchul/cvlface_arcface_ir101_webface4m
