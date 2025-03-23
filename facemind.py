import cv2
import os
from insightface.app import FaceAnalysis
from camera.video_capture import get_video
from face_process.init_InsightFace import Init_model
from face_process.face_recognize import process_frame
from SQL.faces_enroll import enroll_from_camera, enroll_from_image
from SQL.load_known_faces import load_known_faces
from UI.front_end import web_interface

'''
    主函数，整合摄像头调用、人脸识别和数据库加载等所有模块
'''

# 主函数：通过web界面调用完整的FaceMind系统
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
    
    # 从指定的数据库文件中加载已知人脸的特征向量和姓名
    known_face_encodings, known_face_names = load_known_faces(database_path)
    
    # 人脸录入
    # enroll_from_camera(app,'Justin', database_path)
    # enroll_from_image(app, image_path, database_path)
    
    # # 调用本地摄像头进行人脸识别
    # recognize_faces_by_local(app,
    #                          known_face_encodings, 
    #                          known_face_names, 
    #                          threshold)
    
    # 启动Web界面来实现人脸识别和人脸录入
    demo = web_interface(app, 
                         known_face_encodings, 
                         known_face_names, 
                         database_path, 
                         threshold)
    demo.launch()


# 通过OpenCV调用摄像头进行人脸识别（不使用网页UI界面）
def recognize_faces_by_local(app: FaceAnalysis, 
                            known_face_encodings, 
                            known_face_names, 
                            threshold: float=0.5):
    # 通过OpenCV调用本地摄像头实时获取视频帧
    for frame in get_video():
        # 处理视频帧，进行人脸识别
        frame = process_frame(app, 
                              frame, 
                              known_face_encodings,
                              known_face_names, 
                              threshold)
        
        cv2.imshow('Recoginzation', frame)
        
        # 按下Esc键退出，27 是 Esc 键的 ASCII 码
        # 和 0xFF 按位与是为了确保只保留最低 8 位，即ASCII码，以保证不同操作系统的兼容性
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()