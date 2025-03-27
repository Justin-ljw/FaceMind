# @Author        : Justin Lee
# @Time          : 2025-3-27

import os
import insightface.model_zoo as model_zoo
from insightface.app import FaceAnalysis

'''
    根据指定的RetinaFace和ArcFace模型路径，初始化InsightFace模型
    如果没有指定，则使用InsightFace默认的RetinaFace和ArcFace模型
'''


# 初始化InsightFace模型
def Init_model(retinaface_model_path: str=None,
               arcface_model_path: str=None) -> FaceAnalysis:
    # 设置模型的存放位置
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取向上两级目录（到达项目根目录 FaceMind）
    root = os.path.join(parent_directory, '.insightface')  # 构建.insightface 文件夹的路径

    print(root)

    # 推理时优先使用GPU，没有GPU则使用CPU
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=root)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 加载自己微调的RetinaFace和ArcFace模型（如果没有指定，则使用InsightFace默认的模型）
    if retinaface_model_path:
        # 从模型文件加载模型，如：retinaface_model_path = 'path/to/your/retinaface.onnx'
        app.models['detection'] = model_zoo.get_model(retinaface_model_path)
        # 准备模型，即配置模型的上下文设备、阈值、输入尺寸等
        app.models['detection'].prepare(ctx_id=0, input_size=(640, 640))
    
    if arcface_model_path:
        app.models['recognition'] = model_zoo.get_model(arcface_model_path)
        app.models['recognition'].prepare(ctx_id=0)
        
    return app
