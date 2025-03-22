import gradio as gr
import cv2
import os
import sqlite3
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
def enroll_faces_from_camera(app:FaceAnalysis, name: str, connection: sqlite3.Connection):
    try:
        frame, have_face = enroll_from_camera(app, name, connection)
        
        if not have_face:
            return None, "录入失败：未检测到人脸"
        
        return frame, f"人脸 {name} 已成功录入数据库！"
    
    except Exception as e:
        return None, f"录入失败：{str(e)}"


# 上传图片录入界面：通过上传照片录入人脸
def enroll_faces_from_image(app: FaceAnalysis, name: str, image, connection: sqlite3.Connection):
    try:
        image_path = 'temp_image.jpg'
        cv2.imwrite(image_path, image)
        
        frame, have_face = enroll_from_image(app, name, image_path, connection)
        
        os.remove(image_path)
        
        if not have_face:
            return None, "录入失败：未检测到人脸"
        
        return frame, f"人脸 {name} 已成功录入数据库！"
    
    except Exception as e:
        return None, f"录入失败：{str(e)}"


# 主界面：人脸识别界面
def web_interface(app: FaceAnalysis, 
                  known_face_encodings, 
                  known_face_names,   
                  connection: sqlite3.Connection, 
                  threshold: float=0.5):
    
    with gr.Blocks() as demo:
        gr.Markdown("# FaceMind 实时人脸识别系统")
        
        # 实时人脸识别标签页
        with gr.Tab("实时人脸识别"):
            gr.Markdown("## 调用摄像头实现人脸识别")
            
            video_feed = gr.Video(source="webcam", streaming=True)
            # 当视频流变化时调用 recognize_faces_from_camera 函数
            video_feed.change(recognize_faces_from_camera, 
                              inputs=[video_feed, 
                                      gr.State(app), 
                                      gr.State(known_face_encodings), 
                                      gr.State(known_face_names), 
                                      gr.State(threshold)], 
                              outputs=video_feed)
            
            # 跳转到人脸录入界面的按钮
            gr.Button("跳转到人脸录入界面").click(lambda: gr.redirect("/enroll"))
            
            # 人脸录入标签页
        with gr.Tab("人脸录入"):
            gr.Markdown("## 选择人脸录入方式")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 通过摄像头录入人脸")
                    
                    '''
                        这里应该有一个gr.Video组件，代替opencv调用摄像头
                    '''
                    
                    name_input = gr.Textbox(label="请输入姓名")
                    # 开始录入按钮，调用 enroll_faces_from_camera 函数
                    gr.Button("开始录入").click(enroll_faces_from_camera, 
                                            inputs=[gr.Stae(app), 
                                                    name_input, 
                                                    gr.State(connection)], 
                                            outputs=["image", "text"])
                
                with gr.Column():
                    gr.Markdown("### 通过上传照片录入人脸")
                    
                    image_input = gr.Image(source="upload", type="numpy")
                    name_input = gr.Textbox(label="请输入姓名")
                    
                    # 上传并录入按钮，调用 enroll_faces_from_image 函数
                    gr.Button("上传并录入").click(enroll_faces_from_image, 
                                             inputs=[gr.State(app), 
                                                     name_input, 
                                                     image_input, 
                                                     gr.State(connection)], 
                                             outputs=["image", "text"])
            
            # 返回首页按钮
            gr.Button("返回首页").click(lambda: gr.redirect("/"))
            # 继续录入按钮
            gr.Button("继续录入").click(lambda: gr.redirect("/enroll"))
            
    return demo


# 启动Gradio应用
if __name__ == "__main__":
    demo = web_interface()
    demo.launch()