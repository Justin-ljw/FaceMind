import cv2
import sqlite3
import os
from insightface.app import FaceAnalysis
from camera.video_capture import get_video

'''
    人脸录入：把摄像头捕捉到的人脸，通过InsightFace提取特征向量，并存入数据库
'''

def create_database(database_path: str=None):
    
    # 连接到指定路径的SQLite数据库（如果本来没有这个数据库文件，则会自动创建）
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    # 创建faces表，存储人脸图像、姓名和特征向量
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB NOT NULL,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
    ''')
        
    connection.commit()
    connection.close()


# 把人脸特征向量和名字添加到数据库
def add_face_to_database(name, encoding, image, database_path: str=None):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    
    cursor.execute("INSERT INTO faces (image, name, encoding) VALUES (?, ?, ?)", (image, name, encoding.tobytes()))
    connection.commit()
    connection.close()

    
# 处理检测到的人脸
def process_faces(name: str, frame, faces, database_path: str=None):
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
        
        # 将人脸特征向量、名字和人脸图像添加到数据库
        add_face_to_database(name, embedding, face_image_bytes, database_path)
    
    # 关闭窗口
    cv2.destroyAllWindows()
    
    # 判断是否检测到人脸
    have_face = len(faces) > 0
    
    return frame, have_face
    

# 通过摄像头录入人脸
def enroll_from_camera(app: FaceAnalysis, name: str, database_path: str=None):
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    
    # 检查数据库文件是否存在，如果不存在则初始化数据库
    if not os.path.exists(database_path):
        create_database(database_path)
    
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
    return process_faces(name, frame, faces, database_path)
    

# 通过上传图片录入人脸
def enroll_from_image(app: FaceAnalysis, name: str, image_path: str, database_path: str=None):
    assert database_path is not None, 'Error: 请指定数据库文件路径！'
    
    # 检查数据库文件是否存在，如果不存在则初始化数据库
    if not os.path.exists(database_path):
        create_database(database_path)
    
    # 读取图像
    frame = cv2.imread(image_path)
    
    # 使用检测模型检测人脸
    faces = app.get(frame)
    
    # 处理检测到的所有人脸，并返回处理后的视频帧和是否检测到人脸
    return process_faces(name, frame, faces, database_path)   
    