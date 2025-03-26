import cv2
import os
from insightface.app import FaceAnalysis
from merge.mode import User_Mode
from camera.video_capture import get_video
from face_process.init_InsightFace import Init_model
from face_process.face_recognize import process_frame
from SQL.database_operate import create_database
from face_process.faces_enroll import enroll_from_camera_local
from SQL.database_operate import load_known_faces
from UI.front_end import web_interface

'''
    主函数：
    整合本地摄像头调用、人脸识别、数据库加载和前端Web界面等所有模块
'''


# 主函数：通过web界面调用完整的FaceMind系统
def main(mode: User_Mode=User_Mode.WEB, 
         retinaface_model_path: str=None, 
         arcface_model_path: str=None, 
         database_path: str='databases/known_faces.db', 
         gradio_temp_dir: str='gradio_temp/',
         threshold: float=0.6):
    
    # 将相对路径转换为绝对路径（这样就可以通过使用相对路径而兼容不同环境）
    if retinaface_model_path:
        retinaface_model_path = os.path.abspath(retinaface_model_path)
    
    if arcface_model_path:
        arcface_model_path = os.path.abspath(arcface_model_path)
    
    # 数据库文件路径不能为空
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    database_path = os.path.abspath(database_path)
    
    # 检查数据库文件是否存在，如果不存在则初始化数据库
    if not os.path.exists(database_path):
        create_database(database_path)
        
    
    # 通过环境变量设置gradio的临时文件夹路径
    os.environ["GRADIO_TEMP_DIR"] = os.path.abspath(gradio_temp_dir)
    
    
    # 初始化InsightFace模型
    app = Init_model(retinaface_model_path, arcface_model_path)

    
    # web界面模式：可进行视频人脸识别和照片人脸录入
    if mode == User_Mode.WEB:
        # 启动Web界面来实现人脸识别和人脸录入
        demo = web_interface(app,
                            database_path, 
                            threshold)
        demo.launch()
    
    # 本地录入模式：可进行本地摄像头的人脸录入
    elif mode == User_Mode.LOCAL_ENROLL:
        # 通过本地摄像头进行人脸录入
        enroll_from_camera_local(app, database_path)
    
    # 本地识别模式：可进行本地摄像头实时人脸识别
    elif mode == User_Mode.LOCAL_RECOGNIZE:
        # 调用本地摄像头进行人脸识别
        recognize_faces_by_local(app,
                                 database_path,
                                 threshold)


# 通过OpenCV调用摄像头进行人脸识别（不使用网页UI界面）
def recognize_faces_by_local(app: FaceAnalysis,
                             database_path: str,
                             threshold: float=0.6):
    # 从指定的数据库文件中加载已知人脸的特征向量和姓名
    known_face_encodings, known_face_names = load_known_faces(database_path)

    # 通过OpenCV调用本地摄像头实时获取视频帧
    for frame in get_video():
        # 处理视频帧，进行人脸识别
        frame = process_frame(app, 
                              frame, 
                              known_face_encodings,
                              known_face_names, 
                              threshold)
        
        cv2.imshow('FaceMind: Recognize (Esc To Exit)', frame)
        
        # 按下Esc键退出，27 是 Esc 键的 ASCII 码
        # 和 0xFF 按位与是为了确保只保留最低 8 位，即ASCII码，以保证不同操作系统的兼容性
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        
    cv2.destroyAllWindows()
