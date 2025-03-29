# @Author        : Justin Lee
# @Time          : 2025-3-27

import os
import modelscope
import insightface.model_zoo as model_zoo
from insightface.app import FaceAnalysis

'''
    根据指定的RetinaFace和ArcFace模型路径，初始化InsightFace模型
    如果没有指定，则使用InsightFace默认的RetinaFace和ArcFace模型
'''


model_path = {"retinaface": {"repo": None, 
                             "file": None}, 
              "arcface": {"repo": "JustinLeee/FaceMind_ArcFace_iResNet50_CASIA_FaceV5", 
                          "file": "ArcFace_iResNet50_CASIA_FaceV5.onnx"}}


# 如果模型文件不存在，则从 ModelScope 下载模型文件。
def download_model(model_path: str, 
                   repo_name: str, 
                   file_name: str, 
                   local_dir: str):
    print(f"\n模型文件 {model_path} 不存在，正在从 ModelScope 仓库 {repo_name} 下载 {file_name}...")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # 确保目标目录存在

    # 使用 ModelScope API 下载模型文件
    download_path = modelscope.snapshot_download(repo_name, 
                                                 allow_file_pattern=[file_name], 
                                                 local_dir=local_dir)

    print(f"\n模型文件已下载到 {download_path}")


# 初始化InsightFace模型
def Init_model(retinaface_model_path: str=None,
               arcface_model_path: str='.insightface/models/ArcFace_iResNet50_CASIA_FaceV5.onnx') -> FaceAnalysis:
    # 设置模型的存放位置
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取向上两级目录（到达项目根目录 FaceMind）
    root = os.path.join(parent_directory, '.insightface')  # 构建.insightface 文件夹的路径

    print(f"\n模型文件根目录：{root}\n")

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
        # 如果 ArcFace 模型不在指定路径，就从modelscope下载
        if not os.path.exists(arcface_model_path):
            download_model(
                arcface_model_path,
                repo_name=model_path['arcface']['repo'],
                file_name=model_path['arcface']['file'], 
                local_dir=os.path.join(root, 'models')
            )
        
        app.models['recognition'] = model_zoo.get_model(arcface_model_path)
        app.models['recognition'].prepare(ctx_id=0)
        
    return app
