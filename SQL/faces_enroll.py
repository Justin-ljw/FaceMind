import cv2
from insightface.app import FaceAnalysis
from SQL.database_operate import check_name_unique, add_face_to_database
from camera.video_capture import get_video

'''
    人脸录入：
    把摄像头捕捉到的人脸，通过 InsightFace 提取特征向量，并存入数据库
'''
    
# 录入检测到的人脸，录入只支持一个图像一个人脸
def login_face(name: str, 
               frame, 
               face, 
               database_path: str):
    # 提取人脸特征向量
    embedding = face.embedding
    
    # 截取人脸区域子图像
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    face_image = frame[y1:y2, x1:x2]
    
    # 将人脸图像转换为 JPEG 格式的二进制数据
    flag, face_image_encoded = cv2.imencode('.jpg', face_image)
    if flag:
        face_image_bytes = face_image_encoded.tobytes()
    else:
        face_image_bytes = None
        print("Error: 无法将人脸图像转换为 JPEG 格式！")
    
    # 将人脸特征向量、名字和人脸图像添加到数据库
    add_face_to_database(name, 
                         embedding, 
                         face_image_bytes, 
                         database_path)
    
    # 输出检测到的人脸目标框到原始图像
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return frame
    

# 通过本地摄像头录入人脸
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
        
        cv2.imshow('Video', frame)
        
        # 按下Esc键退出
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
        
        # 如果连续检测到10个视频帧都有人脸，则停止拍摄
        if frame_count_have_face >= 10:
            break
    
    '''
        一定是连续检测到了10帧都有人脸才会退出循环，
        所以不存在摄像头检测不到人脸的情况
    '''
    
    # 录入只支持一个图像一个人脸
    name = input('请输入您的姓名：')
    
    # 如果姓名已存在，提示并直接返回
    if check_name_unique(name, database_path):
        print("录入失败：该姓名已存在")
        return None, False
    
    # 录入检测到的人脸
    res = login_face(name, 
                     frame, 
                     faces[0], 
                     database_path)
    
    # 显示检测到的人脸
    cv2.imshow('Get_Face', res)
    # 等待用户按下任意键
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    # 返回处理后的视频帧和是否检测到人脸
    return res, True
    

# 通过照片录入人脸
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
    res = frame
    
    # 如果检测到了人脸就录入
    if have_faces:
        # 处理检测到的所有人脸，并返回处理后的视频帧和是否检测到人脸
        res = login_face(name, 
                         frame, 
                         faces[0], 
                         database_path)
        
    return res, have_faces
    