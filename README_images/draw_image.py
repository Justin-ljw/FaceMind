import cv2
import os
from insightface.app import FaceAnalysis

"""
    显示 RetinaFace 进行人脸检测和关键点预测的结果：
    读取 input_path 指定的图片，使用 InsightFace 进行人脸检测和关键点预测，
    将人脸区域的边框和关键点画在图片上，然后保存到 output_path。
"""


def draw_face(input_path: str,
              output_path: str):
    # 设置模型的存放位置
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取向上两级目录（到达项目根目录 FaceMind）
    root = os.path.join(parent_directory, '.insightface')  # 构建.insightface 文件夹的路径
    
    # 初始化 FaceAnalysis
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=root)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 读取原始图像
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{input_path}")

    # 使用 InsightFace 检测人脸
    faces = app.get(img)
    # 绘制人脸框和关键点
    img = app.draw_on(img, faces)
    
    # 保存绘制后的图像
    cv2.imwrite(output_path, img)
    

if __name__ == "__main__":
    input_path = './README_images/sample_cut.jpg'  # 输入图片路径
    output_path = './README_images/sample_kps.jpg'  # 输出图片路径
    
    draw_face(input_path, output_path)
    print(f"绘制后的图片已保存到：{output_path}")


