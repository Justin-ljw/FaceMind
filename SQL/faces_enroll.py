import cv2
import sqlite3
from insightface.app import FaceAnalysis
from camera.video_capture import get_video

'''
    人脸录入：把摄像头捕捉到的人脸，通过InsightFace提取特征向量，并存入数据库
'''

# 把人脸特征向量和名字添加到数据库
def add_face_to_database(name, encoding, image, connection: sqlite3.Connection):
    cursor = connection.cursor()
    
    cursor.execute("INSERT INTO faces (image, name, encoding) VALUES (?, ?, ?)", (image, name, encoding.tobytes()))
    connection.commit()

    
# 处理检测到的人脸
def process_faces(name: str, frame, faces, connection: sqlite3.Connection):
    for face in faces:
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
        
        # 显示检测到的人脸
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Enroll Face', frame)
                
        # 等待用户点击退出
        while True:
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        
        # 将人脸特征向量、名字和人脸图像添加到数据库
        add_face_to_database(name, embedding, face_image_bytes, connection)
    
    # 关闭窗口
    cv2.destroyAllWindows()
    
    # 判断是否检测到人脸
    have_face = faces is not None
    
    return frame, have_face
    

# 通过摄像头录入人脸
def enroll_from_camera(app: FaceAnalysis, name: str, connection: sqlite3.Connection):
    
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
        
        # 如果连续检测到10个视频帧都有人脸，则停止拍摄
        if frame_count_have_face >= 10:
            break
    
    # 处理检测到的所有人脸，并返回处理后的视频帧和是否检测到人脸
    return process_faces(name, frame, faces, connection)
    

# 通过上传图片录入人脸
def enroll_from_image(app: FaceAnalysis, name: str, image_path: str, connection: sqlite3.Connection):
    # 读取图像
    frame = cv2.imread(image_path)
    
    # 使用检测模型检测人脸
    faces = app.get(frame)
    
    # 处理检测到的所有人脸，并返回处理后的视频帧和是否检测到人脸
    return process_faces(name, frame, faces, connection)   
    