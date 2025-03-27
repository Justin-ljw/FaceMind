# @Author        : Justin Lee
# @Time          : 2025-3-27

import cv2
from insightface.app import FaceAnalysis
from SQL.database_operate import add_face_to_database
from camera.video_capture import get_video

'''
    人脸录入：
    把摄像头捕捉到的人脸，通过 InsightFace 提取特征向量，并存入数据库
'''


# 通过本地摄像头录入人脸（录入只支持一个图像一个人脸）
def enroll_from_camera_local(app: FaceAnalysis, 
                             database_path: str):
    
    # 初始化连续检测到人脸的帧数计数器
    frame_count_have_face = 0
    
    # 调用本地摄像头实时获取视频帧
    for frame in get_video():
        # 使用检测模型检测人脸
        faces = app.get(frame)
        
        # 如果检测到人脸，则计数器加1；否则计数器清零
        if faces:
            frame_count_have_face += 1
        else:
            frame_count_have_face = 0
                    
        # 显示处理后的视频帧
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imshow('FaceMind: CapturedFace (Esc To Exit)', frame)
        
        # 按下Esc键退出
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        
        # 如果连续检测到20个视频帧都有人脸，则停止拍摄
        if frame_count_have_face >= 20:
            break
    
    '''
        一定是连续检测到了10帧都有人脸才会退出循环，
        所以不存在摄像头检测不到人脸的情况
    '''
    
    # 录入只支持一个图像一个人脸
    name = input('请输入您的姓名：')
    
    # 录入检测到的人脸
    add_face_to_database(name,
                         faces[0].embedding,
                         database_path)

    # 截取人脸区域子图像
    x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
    # 输出检测到的人脸目标框到原始图像
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 显示检测到的人脸
    cv2.imshow('FaceMind: Enroll (Esc To Exit)', frame)
    # 等待用户按下任意键
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # 返回处理后的视频帧和是否检测到人脸
    return frame, True
    

# 通过照片录入人脸（录入只支持一个图像一个人脸）
def enroll_from_image(app: FaceAnalysis, 
                      name: str, 
                      image_path: str, 
                      database_path: str):
    
    # 读取图像
    frame = cv2.imread(image_path)
    
    # 使用检测模型检测人脸
    faces = app.get(frame)
    
    # 判断是否检测到了人脸
    have_faces = len(faces) > 0
    # 如果检测到了人脸就录入
    if have_faces:
        # 录入检测到的人脸
        add_face_to_database(name,
                             faces[0].embedding,
                             database_path)

        # 截取人脸区域子图像
        x1, y1, x2, y2 = [int(v) for v in faces[0].bbox]
        # 输出检测到的人脸目标框到原始图像
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    return frame, have_faces
    