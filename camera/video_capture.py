import cv2

'''
    通过OpenCV调用本地摄像头来实时获取视频帧
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
