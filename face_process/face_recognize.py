import cv2
import numpy as np
from insightface.app import FaceAnalysis

'''
    通过InsightFace对摄像头捕捉到的视频帧进行人脸识别：
    1.调用RetinaFace进行人脸检测，获取目标框和关键点
    2.调用ArcFace进行人脸特征提取，获取特征向量
    3.计算当前人脸和已知人脸库的相似度，来识别当前人脸的姓名
    4.把识别结果画在视频帧上，并返回
'''

# 计算cos相似度
def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b)
    return np.dot(a, b) / (a_norm * b_norm)


# 识别人脸
def recognize_faces(app: FaceAnalysis, 
                    frame, 
                    known_face_encodings, 
                    known_face_names, 
                    threshold: float=0.5) -> list:
    # 使用检测模型检测人脸
    faces = app.get(frame)
    
    # 和数据库已知人脸进行匹配，返回人脸框和姓名
    face_names = []
    for face in faces:
        # 未录入的人脸对应的姓名为 Unknown
        name = "Unknown"
        
        # 把当前检测到的人脸和已知人脸库进行匹配
        if known_face_encodings:
            # 计算当前人脸和整个已知人脸矩阵的cos相似度（以矩阵为单位来计算，加速匹配）
            similarities = cosine_similarity(known_face_encodings, face.embedding)
            # 取最大相似度
            max_similarity = np.max(similarities)
            
            # 根据相似度阈值，判断是否为已知人脸
            if max_similarity > threshold:
                index = np.argmax(similarities)
                name = known_face_names[index]
                
        face_names.append((face.bbox, name))
        
    return face_names

# 把人脸识别的结果画在视频帧上
def process_frame(app: FaceAnalysis, 
                  frame, 
                  known_face_encodings,
                  known_face_names, 
                  threshold: float=0.5):
    
    # 识别人脸，返回人脸框和姓名
    face_names = recognize_faces(app, 
                                 frame, 
                                 known_face_encodings, 
                                 known_face_names, 
                                 threshold)
    
    # 在原始视频帧上画出人脸框和姓名
    for (bbox, name) in face_names:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame
