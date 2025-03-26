import gradio as gr
import cv2
import os
from insightface.app import FaceAnalysis
from face_process.face_recognize import process_frame
from SQL.faces_enroll import enroll_from_image
from SQL.database_operate import load_known_faces

'''
    前端Web界面：
    基于Gradio实现一个前端Web界面，
    在此界面，用户可以拍摄/上传视频进行人脸识别、拍摄/上传照片进行人脸录入
'''


# 人脸识别界面：调用摄像头实现人脸识别，当gradio通过摄像头拍到的视频流发生变化时，会回调这个函数
def recognize_faces_from_video(input_path, 
                               app,
                               database_path: str,
                               threshold=0.6):
    # 如果输入地址为None的话，即输入Video的操作是关闭视频，直接输出None，让输出Video的视频也关闭
    if input_path is None:
        return None

    # 从指定的数据库文件中加载已知人脸的特征向量和姓名
    known_face_encodings, known_face_names = load_known_faces(database_path)

    print(input_path)
    
    # 通过输入视频文件的路径，获取输出视频文件的路径
    directory = os.path.dirname(input_path)  # 获取文件所在的目录
    output_filename = 'processed.webm'  # 新的文件名
    output_path = os.path.join(directory, output_filename)  # 新的文件路径
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    
    # 获取视频的帧率、宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # webm视频文件编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # 创建输出视频
    
    while cap.isOpened():
        ret, frame = cap.read()  # 读取一帧
        
        # 读完视频则退出
        if not ret:
            break
        
        # 处理当前帧
        processed_frame = process_frame(app, 
                                        frame, 
                                        known_face_encodings, 
                                        known_face_names, 
                                        threshold)
        
        # 将处理后的帧写入输出视频
        out.write(processed_frame)
        
    # 释放资源
    cap.release()
    out.release()
    
    return output_path


# 通过拍摄或上传照片录入人脸
def enroll_faces_from_image(app: FaceAnalysis, 
                            name: str, 
                            image, 
                            database_path: str):
    if name == "":
        return None, "录入失败：姓名不能为空"
    
    try:
        image_path = 'temp_image.jpg'
        
        # OpenCV 使用 BGR 顺序，而大多数图像处理库使用 RGB 顺序
        # 将图像从 RGB 转换为 BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr)
        
        frame, have_face = enroll_from_image(app, name, image_path, database_path)
        
        os.remove(image_path)
        
        if not have_face:
            return frame, "录入失败：未检测到人脸"
        
        # 将图像从 BGR 转换为 RGB再返回
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"人脸 {name} 已成功录入数据库！"
    
    except Exception as e:
        return None, f"录入失败：{str(e)}"


# 主界面：人脸识别界面
def web_interface(app: FaceAnalysis,
                  database_path: str, 
                  threshold: float=0.6):
    
    with gr.Blocks() as demo:
        gr.Markdown("# FaceMind 人脸识别系统")
        
        # 实时人脸识别标签页
        with gr.Tab("人脸识别"):
            gr.Markdown("## 选择人脸识别方式")
            
            with gr.Tab("拍摄视频"):
                gr.Markdown("## 拍摄视频进行人脸识别")
                
                with gr.Row():
                    with gr.Column():
                        # 负责采集的摄像头
                        video_feed = gr.Video(label='拍摄到的视频', sources="webcam", streaming=True)
                        
                    with gr.Column():
                        # 显示处理后的视频
                        processed_video = gr.Video(label='处理后的视频')
                
                # 当拍摄完视频时调用recognize_faces_from_video函数，将处理后的视频输出到processed_video
                video_feed.change(fn=lambda video_path: recognize_faces_from_video(video_path, 
                                                                            app, 
                                                                            database_path,
                                                                            threshold), 
                                inputs=video_feed, 
                                outputs=processed_video)

                # 刷新当前界面的按钮，以实现再次识别
                def refresh_recognize():
                    return gr.update(value=None), gr.update(value=None)

                gr.Button("再次识别").click(refresh_recognize,
                                            outputs=[video_feed, processed_video])

            with gr.Tab("上传视频"):
                gr.Markdown("## 上传视频进行人脸识别")
                
                with gr.Row():
                    with gr.Column():
                        # 获取上传的视频
                        video_feed = gr.Video(label='上传的视频', sources="upload", streaming=True)
                        
                    with gr.Column():
                        # 显示处理后的视频
                        processed_video = gr.Video(label='处理后的视频')
                        
                # 开始录入按钮
                start = gr.Button("开始人脸识别")
                
                # 点击录入按钮时，调用 enroll_faces_from_camera 函数
                start.click(fn=lambda video_path: recognize_faces_from_video(video_path, 
                                                                        app, 
                                                                        database_path,
                                                                        threshold), 
                            inputs=video_feed, 
                            outputs=processed_video)

                # 刷新当前界面的按钮，以实现再次识别
                gr.Button("再次识别").click(refresh_recognize,
                                            outputs=[video_feed, processed_video])

            
        # 人脸录入标签页
        with gr.Tab("人脸录入"):
            gr.Markdown("## 选择人脸录入方式")
            
            with gr.Tab("拍摄照片"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 通过拍摄照片录入人脸")
                        
                        image_input = gr.Image(sources="webcam", streaming=False)
                        name_input = gr.Textbox(label="请输入姓名")
                        
                    with gr.Column():
                        gr.Markdown("### 录入结果")
                        
                        output_image = gr.Image()
                        output_text = gr.Textbox()

                # 开始录入按钮：点击按钮时，调用 enroll_faces_from_camera 函数
                gr.Button("开始录入").click(lambda image, name: enroll_faces_from_image(app, name, image, database_path),
                                                inputs=[image_input, name_input],
                                                outputs=[output_image, output_text])
                
                # 刷新当前界面的按钮，以实现继续录入
                def refresh_enroll():
                    return gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value="")

                gr.Button("继续录入").click(refresh_enroll, 
                                        outputs=[image_input, name_input, output_image, output_text])

            with gr.Tab("上传照片"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 通过上传照片录入人脸")
                        
                        image_input = gr.Image(sources="upload", type="numpy")
                        name_input = gr.Textbox(label="请输入姓名")
                
                    with gr.Column():
                        gr.Markdown("### 录入结果")
                        
                        output_image = gr.Image()
                        output_text = gr.Textbox()

                # 上传并录入按钮：点击按钮时，调用 enroll_faces_from_image 函数
                gr.Button("上传并录入").click(lambda image, name: enroll_faces_from_image(app, name, image, database_path),
                                                inputs=[image_input, name_input], 
                                                outputs=[output_image, output_text])

                # 刷新当前界面的按钮
                gr.Button("继续录入").click(refresh_enroll, 
                                        outputs=[image_input, name_input, output_image, output_text])

    return demo
