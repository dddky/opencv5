import cv2
import numpy as np
import time
from mss import mss
import os
import torch
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt
from screeninfo import get_monitors

CONFIG_FILE = 'config/screen_config.pkl'
DEFAULT_REGION = {"top": 0, "left": 0, "width": 800, "height": 600}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# 初始化MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

# 手动加载FaceNet模型
FACENET_MODEL_PATH = 'models/20180402-114759-vggface2.pt'
if not os.path.exists(FACENET_MODEL_PATH):
    print(f"错误: FaceNet模型文件不存在: {FACENET_MODEL_PATH}")
    print("请从以下地址下载模型文件并放入models目录:")
    print("https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt")
    exit(1)

try:
    # 尝试直接加载模型而不移除分类层
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print("FaceNet模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    print("尝试替代加载方法...")

    try:
        # 使用更简单的加载方法
        resnet = InceptionResnetV1(classify=False).eval().to(device)
        state_dict = torch.load(FACENET_MODEL_PATH, map_location=device)

        # 适配状态字典键名
        state_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in state_dict.items()}

        # 加载权重
        resnet.load_state_dict(state_dict)
        print("FaceNet模型替代加载成功!")
    except Exception as e2:
        print(f"替代加载方法也失败: {str(e2)}")
        print("无法加载FaceNet模型，程序退出")
        exit(1)


def get_screen_info():
    monitors = get_monitors()
    screen_info = {}

    for i, monitor in enumerate(monitors):
        screen_info[f"Screen_{i + 1}"] = {
            "width": monitor.width,
            "height": monitor.height,
            "x": monitor.x,
            "y": monitor.y
        }

    return screen_info


# 创建目录
def create_directories():
    """创建必要的目录"""
    os.makedirs('config', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)


# 框选屏幕区域
def select_screen_region():
    """让用户框选屏幕区域"""
    # 获取屏幕信息
    screen_info = get_screen_info()
    screens = list(screen_info.keys())

    # 如果没有屏幕信息，使用默认区域
    if not screens:
        print("无法获取屏幕信息，使用默认区域")
        return DEFAULT_REGION

    # 显示屏幕选择菜单
    print("\n" + "=" * 50)
    print("请选择要捕获的屏幕:")
    for i, screen in enumerate(screens):
        info = screen_info[screen]
        print(f"{i + 1}. {screen} ({info['width']}x{info['height']})")
    print("=" * 50)

    # 获取用户选择
    choice = input(f"请选择(1-{len(screens)}): ").strip()
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(screens):
            selected_screen = screens[choice_idx]
            screen_data = screen_info[selected_screen]
            print(f"已选择: {selected_screen}")
        else:
            print("无效选择，使用第一块屏幕")
            selected_screen = screens[0]
            screen_data = screen_info[selected_screen]
    except:
        print("无效输入，使用第一块屏幕")
        selected_screen = screens[0]
        screen_data = screen_info[selected_screen]

    # 创建全屏截图用于框选
    with mss() as sct:
        monitor = {
            "top": screen_data['y'],
            "left": screen_data['x'],
            "width": screen_data['width'],
            "height": screen_data['height']
        }
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 显示框选界面
    cv2.namedWindow("Select Capture Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Capture Region", 1200, 800)
    cv2.putText(img, "请框选人脸捕获区域 (按Enter确认, ESC取消)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 框选区域
    roi = cv2.selectROI("Select Capture Region", img, False)
    cv2.destroyWindow("Select Capture Region")

    if roi == (0, 0, 0, 0):
        print("未选择区域，使用默认设置")
        return DEFAULT_REGION

    x, y, w, h = roi
    capture_region = {
        "top": int(y + screen_data['y']),
        "left": int(x + screen_data['x']),
        "width": int(w),
        "height": int(h)
    }

    print(f"已选择捕获区域: 左上角({capture_region['left']}, {capture_region['top']}) "
          f"尺寸({capture_region['width']}x{capture_region['height']})")

    return capture_region


# 获取捕获区域
def get_capture_region():
    """获取捕获区域配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'rb') as f:
            return pickle.load(f)
    return select_screen_region()


# 人脸注册系统（框选区域捕获）
def face_registration_system():
    """人脸注册系统（自定义区域捕获）"""
    print("=" * 50)
    print("人脸注册系统")
    print("=" * 50)

    name = input("请输入要注册的人名: ").strip()
    if not name:
        print("无效输入")
        return

    # 创建人物目录
    person_dir = os.path.join('dataset', name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"准备注册: {name}")
    print(f"请确保人脸在捕获区域内")

    # 存储所有捕获的区域
    captured_regions = []

    # 捕获前5张图像
    for i in range(5):
        print(f"\n请为第 {i + 1} 张图像框选捕获区域...")
        capture_region = select_screen_region()
        captured_regions.append(capture_region)

        # 捕获选定区域
        with mss() as sct:
            screenshot = sct.grab(capture_region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 保存人脸图像
        save_path = os.path.join(person_dir, f"{name}_{i + 1}.jpg")
        cv2.imwrite(save_path, img)
        print(f"已保存: {save_path}")

        # 检测人脸并提取特征
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # 检测人脸
            boxes, _ = mtcnn.detect(img_pil)

            if boxes is not None:
                # 提取人脸特征（使用最大人脸）
                face_img = mtcnn(img_pil)

                if face_img is not None:
                    # 处理多张人脸情况
                    if face_img.dim() == 4:  # 多张人脸
                        face_img = face_img[0]  # 取第一个检测到的人脸

                    embeddings = resnet(face_img.unsqueeze(0).to(device))
                    embedding = embeddings.detach().cpu().numpy()[0]

                    # 保存特征向量
                    np.save(save_path.replace('.jpg', '.npy'), embedding)
                    print(f"人脸特征已保存")
        except Exception as e:
            print(f"捕获失败: {str(e)}")

    # 询问用户是否继续上传更多图像
    additional_count = 5
    while True:
        cont = input("\n是否继续上传更多图像? (y/n): ").strip().lower()
        if cont == 'n':
            break
        elif cont == 'y':
            print(f"\n请为第 {additional_count + 1} 张图像框选捕获区域...")
            capture_region = select_screen_region()
            captured_regions.append(capture_region)

            # 捕获选定区域
            with mss() as sct:
                screenshot = sct.grab(capture_region)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # 保存人脸图像
            save_path = os.path.join(person_dir, f"{name}_{additional_count + 1}.jpg")
            cv2.imwrite(save_path, img)
            print(f"已保存: {save_path}")

            # 检测人脸并提取特征
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                # 检测人脸
                boxes, _ = mtcnn.detect(img_pil)

                if boxes is not None:
                    # 提取人脸特征（使用最大人脸）
                    face_img = mtcnn(img_pil)

                    if face_img is not None:
                        # 处理多张人脸情况
                        if face_img.dim() == 4:  # 多张人脸
                            face_img = face_img[0]  # 取第一个检测到的人脸

                        embeddings = resnet(face_img.unsqueeze(0).to(device))
                        embedding = embeddings.detach().cpu().numpy()[0]

                        # 保存特征向量
                        np.save(save_path.replace('.jpg', '.npy'), embedding)
                        print(f"人脸特征已保存")
            except Exception as e:
                print(f"捕获失败: {str(e)}")

            additional_count += 1
        else:
            print("请输入 y 或 n")

    # 保存本次捕获的所有区域配置
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump(captured_regions[-1], f)  # 保存最后一个区域作为默认

    print(f"{name} 注册完成! 共捕获 {additional_count} 张图像")


# 训练人脸识别模型（FaceNet + KNN）
def train_face_recognition_model():
    """训练人脸识别模型"""
    print("训练人脸识别模型...")
    start_time = time.time()

    # 加载数据集
    dataset_path = 'dataset'
    embeddings = []
    labels = []

    # 遍历所有人物目录
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_dir):
            continue

        # 遍历每个人的特征文件
        for file_name in os.listdir(person_dir):
            if file_name.endswith('.npy'):
                embedding_path = os.path.join(person_dir, file_name)

                try:
                    emb = np.load(embedding_path)
                    embeddings.append(emb)
                    labels.append(person_name)
                except Exception as e:
                    print(f"加载特征出错 {embedding_path}: {str(e)}")

    # 检查是否有足够数据
    if len(embeddings) < 2:
        print("错误: 数据集不足，请先注册至少两个人的人脸")
        return None, None

    # 转换数据为numpy数组
    X = np.array(embeddings)
    y = np.array(labels)

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    knn.fit(X, y_encoded)

    # 保存模型
    model_path = 'models/face_recognition_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((knn, le), f)

    elapsed_time = time.time() - start_time
    print(f"训练完成! 共训练 {len(embeddings)} 个人脸特征")
    print(f"注册人员: {', '.join(le.classes_)}")
    print(f"训练耗时: {elapsed_time:.2f}秒")

    return knn, le


# 实时人脸识别系统（框选区域）
def realtime_face_recognition():
    """实时人脸识别系统（自定义区域）"""
    print("\n请框选识别区域...")
    capture_region = select_screen_region()

    # 加载模型
    model_path = 'models/face_recognition_model.pkl'

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                knn, le = pickle.load(f)
            print(f"加载识别模型，已注册人脸: {len(le.classes_)}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print("请先训练模型")
            return
    else:
        print("未找到训练模型，请先训练模型")
        return

    if knn is None or le is None:
        return

    # 创建窗口
    window_width = min(1200, capture_region['width'])
    window_height = min(800, capture_region['height'])
    cv2.namedWindow("Face Recognition System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition System", window_width, window_height)

    print("=" * 50)
    print(f"人脸识别系统已启动 (区域: {capture_region['width']}x{capture_region['height']})")
    print("按 'q' 退出")
    print("=" * 50)

    frame_count = 0
    start_time = time.time()
    sct = mss()  # 初始化屏幕捕获对象

    # 置信度阈值
    confidence_threshold = 0.7

    while True:
        frame_count += 1
        # 捕获选定区域
        screenshot = sct.grab(capture_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 将BGRA转换为BGR
        display_frame = frame.copy()

        # 显示区域边界
        cv2.rectangle(display_frame, (0, 0),
                      (capture_region['width'], capture_region['height']),
                      (0, 255, 0), 2)

        # 转换为PIL格式进行人脸检测
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # 人脸检测
            boxes, probs = mtcnn.detect(pil_img)

            # 处理检测到的人脸
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    x1, y1, x2, y2 = map(int, box)

                    # 绘制人脸框
                    color = (0, 255, 0) if prob > 0.9 else (0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    # 显示置信度
                    cv2.putText(display_frame, f"Face: {prob:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

                    # 提取人脸区域
                    face_img = mtcnn(pil_img.crop((x1, y1, x2, y2)))

                    if face_img is not None:
                        # 提取特征
                        embeddings = resnet(face_img.unsqueeze(0).to(device))
                        embedding = embeddings.detach().cpu().numpy()[0]

                        # 人脸识别
                        distances, indices = knn.kneighbors([embedding], n_neighbors=1)
                        distance = distances[0][0]

                        # 计算置信度（基于距离）
                        confidence = max(0, min(1, 1.0 - distance))

                        # 获取标签
                        pred_label = le.inverse_transform(indices[0])[0]

                        # 根据置信度显示结果
                        if confidence > confidence_threshold:
                            text = f"{pred_label} ({confidence:.2f})"
                            color = (0, 255, 0)
                        else:
                            text = "Unknown"
                            color = (0, 0, 255)

                        # 显示识别结果
                        cv2.putText(display_frame, text, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print(f"识别错误: {str(e)}")

        # 计算FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 添加系统信息
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"area: {capture_region['width']}x{capture_region['height']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display_frame, "push 'q' back", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 显示结果
        cv2.imshow("Face Recognition System", display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("识别系统已关闭")


# 主菜单系统
def main_menu():
    """主菜单系统"""
    create_directories()

    while True:
        print("\n" + "=" * 50)
        print("人脸识别系统菜单")
        print("=" * 50)
        print("1. 人脸注册（框选区域）")
        print("2. 训练识别模型")
        print("3. 实时人脸识别（框选区域）")
        print("4. 退出")
        print("=" * 50)

        choice = input("请选择操作: ").strip()

        if choice == '1':
            face_registration_system()
        elif choice == '2':
            train_face_recognition_model()
        elif choice == '3':
            realtime_face_recognition()
        elif choice == '4':
            print("感谢使用，再见!")
            break
        else:
            print("无效选择，请重新输入")


# 运行主程序
if __name__ == "__main__":
    main_menu()