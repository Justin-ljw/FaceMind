import cv2
import os
from camera.video_capture import get_video
from face_process.init_InsightFace import Init_model
from face_process.face_recognize import process_frame
from SQL.sqlite_connect import get_connection
from SQL.faces_enroll import enroll_from_camera, enroll_from_image
from SQL.load_known_faces import load_known_faces
from UI.front_end import web_interface

'''
    主函数，整合摄像头调用、人脸识别和数据库加载等所有模块
'''

def main(retinaface_model_path: str=None, 
         arcface_model_path: str=None, 
         database_path: str='databases/known_faces.db', 
         threshold: float=0.5):
    
    # 将相对路径转换为绝对路径（这样就可以通过使用相对路径而兼容不同环境）
    if retinaface_model_path:
        retinaface_model_path = os.path.abspath(retinaface_model_path)
    
    if arcface_model_path:
        arcface_model_path = os.path.abspath(arcface_model_path)
    
    database_path = os.path.abspath(database_path)
    
    
    # 初始化InsightFace模型
    app = Init_model(retinaface_model_path, arcface_model_path)
    
    # 获取数据库连接
    connection = get_connection(database_path)
    
    # 人脸录入
    # enroll_from_camera(app, connection)
    # enroll_from_image(app, image_path, connection)
    
    # 从指定的数据库文件中加载已知人脸的特征向量和姓名
    known_face_encodings, known_face_names = load_known_faces(connection)
    
    # 启动Web界面
    demo = web_interface(app, 
                         known_face_encodings, 
                         known_face_names, 
                         connection, 
                         threshold)
    demo.launch()
    
    # 结束时要关闭数据库连接
    connection.close()

if __name__ == "__main__":
    main()