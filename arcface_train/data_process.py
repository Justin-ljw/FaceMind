import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align

'''
    这里的数据处理没有进行norm，要在训练加载数据时实现
'''


# 定义图像尺寸调整函数
def resize_image(image, target_size=(112, 112)):
    return cv2.resize(image, target_size)


# 定义水平翻转数据增强函数
# def hflip_image(image):
#     flipped_image = cv2.flip(image, 1)
#     return flipped_image


# 定义垂直翻转数据增强函数
def vflip_image(image):
    flipped_image = cv2.flip(image, 0)
    return flipped_image


# 定义逆时针旋转数据增强函数（顺时针的话，只要angle参数为负角度即可）
def rotate(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # 采用边界填充，避免引入大部分黑色区域
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


# 定义调大亮度数据增强函数
def increase_brightness(image, value=60):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    brighter_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return brighter_image


# 定义调小亮度数据增强函数
def decrease_brightness(image, value=40):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v[v < value] = 0
    v[v >= value] -= value
    final_hsv = cv2.merge((h, s, v))
    darker_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return darker_image


# 定义中值模糊数据增强函数
def median_blur_image(image, kernel_size=7):
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


# 标准五官位置
STANDARD_LANDMARKS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


# 黑色矩形遮挡眼睛，模拟墨镜
def block_eyes(image):
    # 提取左右眼坐标
    left_eye = STANDARD_LANDMARKS[0].astype(int)
    right_eye = STANDARD_LANDMARKS[1].astype(int)

    # 计算眼睛矩形边界（适当扩展覆盖范围）
    x_min = max(0, min(left_eye[0], right_eye[0]) - 15)
    x_max = min(111, max(left_eye[0], right_eye[0]) + 15)
    y_min = max(0, min(left_eye[1], right_eye[1]) - 10)
    y_max = min(111, max(left_eye[1], right_eye[1]) + 10)

    # cv2绘制图像会在原图上修改，但是并不想对原图进行修改，所以先复制一份再去绘制遮挡
    image_copy = image.copy()
    # 绘制黑色矩形遮挡眼睛
    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    return image_copy


# 白色圆形遮挡鼻子和嘴巴，模拟口罩
def block_nose_mouth(image):
    # 提取鼻子和嘴巴坐标
    nose = STANDARD_LANDMARKS[2].astype(int)
    mouth_left = STANDARD_LANDMARKS[3].astype(int)
    mouth_right = STANDARD_LANDMARKS[4].astype(int)

    # 计算遮挡圆的中心和半径
    center = ((nose[0] + mouth_left[0] + mouth_right[0]) // 3,
              (nose[1] + mouth_left[1] + mouth_right[1]) // 3)
    radius = int(np.linalg.norm(nose - mouth_left))  # 基于距离计算半径

    # cv2绘制图像会在原图上修改，但是并不想对原图进行修改，所以先复制一份再去绘制遮挡
    image_copy = image.copy()
    # 绘制白色圆形遮挡鼻子和嘴巴
    cv2.circle(image_copy, center, radius, (255, 255, 255), -1)
    return image_copy


"""
该函数用于获取图像的 RGB 三个通道，并生成仅保留单一通道颜色的图像。
:param image: 输入的图像，应为 OpenCV 读取的图像（BGR 格式）
:return: 返回三个数组，分别代表仅含红色、绿色、蓝色的图像
"""
def get_single_color(image, channel: int):
    # 检查输入的通道参数是否为 0、1 或 2
    if channel not in (0, 1, 2):
        raise ValueError("通道参数必须是 0、1、2")

    # 创建空白图像用于存储单一通道的图像
    single_color_image = np.zeros_like(image)
    # 提取单一通道
    single_color_image[:, :, channel] = image[:, :, channel]

    return single_color_image


# 保存图像
def save_image(output_dir, file_name, image):
    output_path = os.path.join(output_dir, f"{file_name}.jpg")
    output_image = cv2.cvtColor(image.astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_image)

    # cv2.imshow("output_image", output_image)
    # while True:
    #     if cv2.waitKey(0):
    #         break


def face_data_process(input_dataset_path: str,
                      output_dataset_path: str):
    os.makedirs(output_dataset_path, exist_ok=True)

    # 设置模型的存放位置
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    parent_directory = os.path.dirname(os.path.dirname(current_file_path))  # 获取向上两级目录（到达项目根目录 FaceMind）
    root = os.path.join(parent_directory, '.insightface')  # 构建.insightface 文件夹的路径

    # 初始化insightface的FaceAnalysis对象，用于人脸检测和对齐
    app = FaceAnalysis(root=root)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 遍历输入数据集中的所有图像文件
    for root, dirs, files in tqdm(os.walk(input_dataset_path)):
        for file in files:
            if file.endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]  # 获取文件名（不含后缀）

                if file.endswith(('.bmp')):  # 修改为判断bmp格式
                    img = cv2.imread(image_path)  # 使用cv2.imread读取bmp图像
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式，insightface要求RGB输入
                else:
                    img = ins_get_image(image_path)

                faces = app.get(img)

                if len(faces) > 0:
                    for face in faces:
                        # 人脸对齐
                        # 获取对齐后的人脸图像，并自动裁剪
                        aligned_face = face_align.norm_crop(img, landmark=face.kps, image_size=112)

                        # 获取输出文件夹路径
                        output_dir = os.path.join(output_dataset_path, os.path.relpath(root, input_dataset_path))
                        os.makedirs(output_dir, exist_ok=True)

                        # 保存原始处理后的图像
                        save_image(output_dir, base_name, aligned_face)

                        '''
                            数据增强
                        '''

                        # # 水平翻转
                        # hflipped_face = hflip_image(aligned_face)
                        # save_image(output_dir, f"{base_name}_hflip", hflipped_face)

                        # 垂直翻转
                        vflipped_face = vflip_image(aligned_face)
                        save_image(output_dir, f"{base_name}_vflip", vflipped_face)

                        # 顺时针旋转45度
                        rotated_clockwise_45_face = rotate(aligned_face, -45)
                        save_image(output_dir, f"{base_name}_left_rotate45", rotated_clockwise_45_face)

                        # 逆时针旋转45度
                        rotated_counterclockwise_45_face = rotate(aligned_face, 45)
                        save_image(output_dir, f"{base_name}_right_rotate_45", rotated_counterclockwise_45_face)

                        # 调大亮度
                        brighter_face = increase_brightness(aligned_face)
                        save_image(output_dir, f"{base_name}_brighter", brighter_face)

                        # 调小亮度
                        darker_face = decrease_brightness(aligned_face)
                        save_image(output_dir, f"{base_name}_darker", darker_face)

                        # 中值模糊
                        blurred_face = median_blur_image(aligned_face)
                        save_image(output_dir, f"{base_name}_blurred", blurred_face)

                        # 黑色矩形遮挡双眼，模拟墨镜
                        block_eyes_face = block_eyes(aligned_face)
                        save_image(output_dir, f"{base_name}_block_eyes", block_eyes_face)

                        # 白色圆形遮挡嘴巴和鼻子，模拟口罩
                        block_mouth_face = block_nose_mouth(aligned_face)
                        save_image(output_dir, f"{base_name}_block_mouth", block_mouth_face)

                        # 分别获取 RGB 三个通道单一颜色的图像
                        red_image = get_single_color(aligned_face, 0)
                        save_image(output_dir, f"{base_name}_red", red_image)

                        green_image = get_single_color(aligned_face, 1)
                        save_image(output_dir, f"{base_name}_green", green_image)

                        blue_image = get_single_color(aligned_face, 2)
                        save_image(output_dir, f"{base_name}_blue", blue_image)

                        # 获取灰度图
                        gray_output_path = os.path.join(output_dir, f"{base_name}_gray.jpg")
                        gray_output_image = cv2.cvtColor(aligned_face.astype(np.uint8),
                                                    cv2.COLOR_RGB2GRAY)
                        cv2.imwrite(gray_output_path, gray_output_image)


if __name__ == '__main__':
    # 输入数据集路径和输出数据集路径
    input_dataset_path = "./origin_dataset/CASIA_FaceV5"  # 替换为你的输入数据集路径
    output_dataset_path = "./processed_dataset/Cleaned_CASIA_FaceV5"  # 替换为你的输出数据集路径

    face_data_process(input_dataset_path, output_dataset_path)
