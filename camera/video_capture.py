import cv2

'''
    通过OpenCV调用摄像头来实时获取视频帧
'''

def get_video():
    # 打开摄像头
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: 无法打开摄像设备！")
        return

    # 设置视频帧的宽度和高度
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        # 捕捉视频帧（捕捉到的图像是以Matlike类型返回的）
        # Matlike类型约等于numpy数组，numpy数组可以直接用于OpenCV的处理
        ret, frame = video_capture.read()

        if not ret:
            print("Error: 无法读取视频帧！")
            break

        # 左右翻转图像，因为摄像头捕捉图像的方向和输出图像的方向是相反的
        frame = cv2.flip(frame, 1)
        
        # 不停将捕捉到的视频帧返回
        yield frame

    # 释放摄像头并关闭所有OpenCV窗口
    video_capture.release()
    cv2.destroyAllWindows()

# 测试
if __name__ == "__main__":
    for frame in get_video():
        cv2.imshow('Video', frame)
        
        # 和 0xFF 按位与是为了确保只保留最低 8 位，即ASCII码，以保证不同操作系统的兼容性
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()