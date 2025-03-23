import gradio as gr
import cv2
import os
from insightface.app import FaceAnalysis
from camera.video_capture import get_video
from face_process.face_recognize import process_frame
from SQL.faces_enroll import enroll_from_camera, enroll_from_image

# 人脸识别界面：调用摄像头实现人脸识别，当gradio通过摄像头拍到的视频流发生变化时，会回调这个函数
def recognize_faces_from_camera(frame, 
                                app: FaceAnalysis,  
                                known_face_encodings,
                                known_face_names, 
                                threshold: float=0.5):
    # 处理视频帧，进行人脸识别
    frame = process_frame(app, 
                          frame, 
                          known_face_encodings, 
                          known_face_names, 
                          threshold)
    
    return frame
        
    #     # 显示处理后的视频帧
    #     cv2.imshow('Video', frame)
        
    #     # 和 0xFF 按位与是为了确保只保留最低 8 位，即ASCII码，以保证不同操作系统的兼容性
    #     if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #         break
        
    # cv2.destroyAllWindows()


# 摄像头录入界面：通过摄像头录入人脸
def enroll_faces_from_camera(app:FaceAnalysis, name: str, database_path: str=None):
    try:
        frame, have_face = enroll_from_camera(app, name, database_path)
        
        if not have_face:
            return None, "录入失败：未检测到人脸"
        
        # 将图像从 BGR 转换为 RGB再返回
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"人脸 {name} 已成功录入数据库！"
    
    except Exception as e:
        return None, f"录入失败：{str(e)}"


# 上传图片录入界面：通过上传照片录入人脸
def enroll_faces_from_image(app: FaceAnalysis, name: str, image, database_path: str=None):
    try:
        image_path = 'temp_image.jpg'
        
        # OpenCV 使用 BGR 顺序，而大多数图像处理库使用 RGB 顺序
        # 将图像从 RGB 转换为 BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr)
        
        frame, have_face = enroll_from_image(app, name, image_path, database_path)
        
        os.remove(image_path)
        
        if not have_face:
            return None, "录入失败：未检测到人脸"
        
        # 将图像从 BGR 转换为 RGB再返回
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"人脸 {name} 已成功录入数据库！"
    
    except Exception as e:
        return None, f"录入失败：{str(e)}"


# 主界面：人脸识别界面
def web_interface(app: FaceAnalysis, known_face_encodings, known_face_names, database_path: str=None, threshold: float=0.5):
    with gr.Blocks() as demo:
        gr.Markdown("# FaceMind 实时人脸识别系统")
        
        # 实时人脸识别标签页
        with gr.Tab("实时人脸识别"):
            gr.Markdown("## 调用摄像头实现人脸识别")
            
            video_feed = gr.Video(sources="webcam", streaming=True)
            # 当视频流变化时调用 recognize_faces_from_camera 函数
            video_feed.change(fn=lambda frame: recognize_faces_from_camera(frame, 
                                                                           app, 
                                                                           known_face_encodings, 
                                                                           known_face_names, 
                                                                           threshold), 
                              inputs=video_feed, 
                              outputs=video_feed)
            
            
        # 人脸录入标签页
        with gr.Tab("人脸录入"):
            gr.Markdown("## 选择人脸录入方式")
            
            with gr.Tab("摄像头录入"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 通过摄像头录入人脸")
                        
                        name_input = gr.Textbox(label="请输入姓名")
                        
                        # 开始录入按钮
                        checkin = gr.Button("开始录入")
                        
                    with gr.Column():
                        gr.Markdown("### 录入结果")
                        
                        output_image = gr.Image()
                        output_text = gr.Textbox()
                
                # 点击录入按钮时，调用 enroll_faces_from_camera 函数
                checkin.click(lambda name: enroll_faces_from_camera(app, name, database_path) if name else "姓名不能为空", 
                                                inputs=name_input,
                                                outputs=[output_image, output_text])
                
                # 刷新当前界面的按钮，以实现继续录入
                def refresh_camera_input():
                    return gr.update(value=""), gr.update(value=None), gr.update(value="")

                gr.Button("继续录入").click(refresh_camera_input, outputs=[name_input, output_image, output_text])
                
            
            with gr.Tab("上传照片录入"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 通过上传照片录入人脸")
                        
                        image_input = gr.Image(sources="upload", type="numpy")
                        name_input = gr.Textbox(label="请输入姓名")
                        
                        # 上传并录入按钮
                        checkin = gr.Button("上传并录入")
                
                    with gr.Column():
                        gr.Markdown("### 录入结果")
                        
                        output_image = gr.Image()
                        output_text = gr.Textbox()

                # 点击录入按钮时，调用 enroll_faces_from_image 函数
                checkin.click(lambda image, name: enroll_faces_from_image(app, name, image, database_path) if name else "姓名不能为空", 
                                                inputs=[image_input, name_input], 
                                                outputs=[output_image, output_text])
                
                # 刷新当前界面的按钮，以实现继续录入
                def refresh_image_input():
                    return gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value="")

                gr.Button("继续录入").click(refresh_image_input, outputs=[image_input, name_input, output_image, output_text])
            

    return demo
