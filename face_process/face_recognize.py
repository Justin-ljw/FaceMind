# @Author        : Justin Lee
# @Time          : 2025-3-27

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

'''
    通过InsightFace对摄像头捕捉到的视频帧进行人脸识别：
    1.调用RetinaFace进行人脸检测，获取目标框和关键点
    2.调用ArcFace进行人脸特征提取，获取特征向量
    3.计算当前人脸和已知人脸库的相似度，来识别当前人脸的姓名
    4.把识别结果画在视频帧上，并返回
'''


# 计算cos相似度（如果设备有GPU的话，可用GPU加速，没有的话使用CPU）
def cosine_similarity(known_faces, current_face, device):
    with torch.no_grad():
        known_faces = torch.tensor(known_faces, dtype=torch.float32).to(device)
        current_face = torch.tensor(current_face, dtype=torch.float32).to(device)

        known_faces_norm = torch.linalg.norm(known_faces, dim=1, keepdims=True).reshape(-1)
        current_face_norm = torch.linalg.norm(current_face, dim=0, keepdims=True)

        cos = torch.matmul(known_faces, current_face) / (known_faces_norm * current_face_norm)

        return cos.cpu().numpy()


# 识别人脸
def recognize_faces(app: FaceAnalysis,
                    frame,
                    known_face_encodings,
                    known_face_names,
                    threshold: float = 0.5,
                    batch_size: int = 1024) -> list:
    # 使用检测模型检测人脸
    faces = app.get(frame)

    # 如果有GPU的话选择GPU进行矩阵运算加速，没有的话使用CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 和数据库已知人脸进行匹配，返回人脸框和姓名
    face_names = []
    max_similarity_overall = 0
    for face in faces:
        # 未录入的人脸对应的姓名为 Unknown
        name = "Unknown"

        # 记录所有已知人脸和当前人脸相似度最大的最大值
        max_similarity_overall = 0

        # 计算数据库中所有的已知人脸有多少个batch
        num_batches = len(known_face_encodings) // batch_size + (1 if len(known_face_encodings) % batch_size != 0 else 0)
        # 按批次处理已知人脸编码
        for i in range(num_batches):
            # 计算当前batch的范围，并截取
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(known_face_encodings))
            batch_encodings = known_face_encodings[start_idx:end_idx]

            # 计算当前人脸和当前批次已知人脸矩阵的cos相似度
            similarities = cosine_similarity(batch_encodings, face.embedding, device)
            # 取当前批次的最大相似度
            max_similarity_batch = np.max(similarities)

            # 更新全局最大相似度和对应的姓名
            if max_similarity_batch > max_similarity_overall:
                max_similarity_overall = max_similarity_batch

                # 根据相似度阈值，判断是否为已知人脸，有已知人脸匹配才更新姓名
                if max_similarity_batch > threshold:
                    index_in_batch = np.argmax(similarities)
                    index = start_idx + index_in_batch
                    name = known_face_names[index]

        print(f"cos_similarity: {max_similarity_overall}")

        face_names.append((face.bbox, name))

    return face_names, max_similarity_overall


# 把人脸识别的结果画在视频帧上
def process_frame(app: FaceAnalysis, 
                  frame, 
                  known_face_encodings,
                  known_face_names, 
                  threshold: float=0.5):
    
    # 识别人脸，返回人脸框和姓名
    face_names, similarity = recognize_faces(app, 
                                 frame, 
                                 known_face_encodings, 
                                 known_face_names, 
                                 threshold)
    
    # 如果没有检测到人脸，直接返回，并标记未检测到
    if len(face_names) <= 0:
        return frame, False, 0

    # 因为 OpenCV 不能绘制中文，所以要将 OpenCV 图像转换为 PIL 图像，绘制完再转回OpenCV图像
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("msyhbd.ttc", 30)  # 请确保系统中有这个字体文件
    
    # 在原始视频帧上画出人脸框和姓名
    for (bbox, name) in face_names:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # 在 PIL 图像上绘制人脸框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        # 在 PIL 图像上绘制中文姓名
        draw.text((x1, y2), name, font=font, fill=(0, 255, 0))
    
    # 将 PIL 图像转换回 OpenCV 图像
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return frame, True, similarity
