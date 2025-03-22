import insightface.model_zoo as model_zoo
from insightface.app import FaceAnalysis

'''
    根据指定的RetinaFace和ArcFace模型路径，初始化InsightFace模型
    如果没有指定，则使用InsightFace默认的RetinaFace和ArcFace模型
'''

# 初始化InsightFace模型
def Init_model(retinaface_model_path: str=None, arcface_model_path: str=None) -> FaceAnalysis:
    # 推理时优先使用GPU，没有GPU则使用CPU
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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