from merge.facemind import main
from merge.mode import User_Mode

'''
    FaceMind 客户端：
    由此可以直接使用 FaceMind 实时人脸识别系统，
    允许用户自主选择模式、模型文件、数据库路径、Gradio临时文件夹路径和人脸识别阈值等,
    若无特殊需求，使用默认值即可
'''

def facemind_client(mode: User_Mode=User_Mode.WEB, 
                    retinaface_model_path: str=None, 
                    arcface_model_path: str=None, 
                    database_path: str='databases/known_faces.db', 
                    gradio_temp_dir: str='gradio_temp/', 
                    threshold: float=0.6):
    main(mode, 
         retinaface_model_path, 
         arcface_model_path, 
         database_path, 
         gradio_temp_dir, 
         threshold)
    
    
if __name__ == "__main__":
    '''
        用户模式：
        1.WEB (Web界面模式)：可进行视频人脸识别和照片人脸录入
        2.LOCAL_ENROLL (本地录入模式)：可进行本地摄像头的人脸录入
        3.LOCAL_RECOGNIZE (本地识别模式)：可进行本地摄像头实时人脸识别
        （LOCAL_ENROLL 和 LOCAL_RECOGNIZE 按 Esc 键退出）
    '''
    mode: User_Mode=User_Mode.WEB
    
    '''
        RetinaFace 和 ArcFace 模型的路径：
        可使用自己的 RetinaFace 和 ArcFace 模型，
        默认都为None，即使用默认的 RetinaFace 和 ArcFace 模型
    '''
    retinaface_model_path: str=None
    arcface_model_path: str=None
    
    '''
        数据库文件路径：
        用于存储录入的人脸信息
    '''
    database_path: str='databases/known_faces.db'
    
    '''
        Gradio的临时文件路径：
        Web界面模式由Gradio实现，会存储上传的图片、视频等文件
    '''
    gradio_temp_dir: str='gradio_temp/'
    
    '''
        人脸识别阈值：
        用于判断要识别的人脸是否已录入，
        threshold数值越大，对匹配的相识度要求越高，即越容易认为不匹配
    '''
    threshold: float=0.6
    
    facemind_client(mode, 
                    retinaface_model_path, 
                    arcface_model_path, 
                    database_path, 
                    gradio_temp_dir, 
                    threshold)