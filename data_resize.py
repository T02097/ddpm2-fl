import os
from PIL import Image
from torchvision.transforms import Resize
from torchvision import datasets, transforms
from tqdm import tqdm


def resize_custom_mnist_dataset(dataset_path, output_dir, target_size=(64, 64)):
    """
    处理自定义MNIST格式数据集并转换为指定尺寸
    参数：
    dataset_path: 自定义数据集的根路径（需包含0-9子文件夹）
    output_dir: 输出目录路径
    target_size: 目标尺寸 (默认64x64)
    """
    # 创建输出目录结构
    for digit in range(10):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    # 定义缩放变换
    resize_transform = Resize(target_size)

    total_count = 0

    # 遍历每个类别文件夹
    for label in sorted(os.listdir(dataset_path)):
        label_dir = os.path.join(dataset_path, label)

        # 跳过非目录文件和无效标签
        if not os.path.isdir(label_dir):
            continue
        if not label.isdigit() or int(label) not in range(10):
            continue

        label_int = int(label)
        img_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # 使用进度条处理每个类别
        for idx, img_file in enumerate(tqdm(img_files, desc=f"处理类别 {label}")):
            try:
                # 加载图像
                img_path = os.path.join(label_dir, img_file)
                image = Image.open(img_path).convert('L')  # 强制转换为灰度图

                # 调整尺寸
                resized_img = resize_transform(image)

                # 生成标准文件名（保留原始索引或使用连续编号）
                filename = f"{idx:05d}.png"
                output_path = os.path.join(output_dir, label, filename)

                # 保存图像
                resized_img.save(output_path, "PNG")
                total_count += 1
            except Exception as e:
                print(f"无法处理 {img_path}: {str(e)}")

    print(f"处理完成！共保存 {total_count} 张图像到 {output_dir}")

def check_image_sizes(root_dir):
    print(f"查看图像路径：{root_dir}")
    sizes = set()
    count = 0

    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.png'):
                img_path = os.path.join(root, f)
                try:
                    img = Image.open(img_path)
                    sizes.add(img.size)
                    count += 1
                except Exception as e:
                    print(f"读取失败: {img_path}, 错误: {e}")

    print(f"\n检查完成，总计 {count} 张图像")
    print(f"发现的尺寸集合: {sizes}")
    if len(sizes) == 1:
        print("✅所有图像尺寸一致!")
    else:
        print("⚠️存在尺寸不一致的图像，请检查!")
    print("-----------------------------------")


if __name__ == "__main__":
    # # 下载原始数据集（你已有就不会重复下载）
    # train_dataset = datasets.FashionMNIST('./data', train=True, download=False)
    # test_dataset = datasets.FashionMNIST('./data', train=False, download=False)

    # # 主目录创建
    # os.makedirs('./fmnist_yuan/train', exist_ok=True)
    # os.makedirs('./fmnist_yuan/test', exist_ok=True)

    # num_classes = 10
    # train_dir = './fmnist_yuan/train'
    # test_dir = './fmnist_yuan/test'

    # # 为每个类别创建子文件夹
    # for split_dir in [train_dir, test_dir]:
    #     for label in range(num_classes):
    #         os.makedirs(os.path.join(split_dir, str(label)), exist_ok=True)

    # # 计数器
    # train_count = {label: 0 for label in range(num_classes)}
    # test_count = {label: 0 for label in range(num_classes)}

    # # 保存训练集
    # for image, label in train_dataset:
    #     save_path = os.path.join(train_dir, str(label), f"{train_count[label]}.png")
    #     image.save(save_path)
    #     train_count[label] += 1

    # # 保存测试集
    # for image, label in test_dataset:
    #     save_path = os.path.join(test_dir, str(label), f"{test_count[label]}.png")
    #     image.save(save_path)
    #     test_count[label] += 1

    # print("✅图片分类保存完成")


    # CUSTOM_DATASET_PATH = "fmnist_yuan/train"  # 包含0-9子文件夹的数据集根路径
    # OUTPUT_DIR = "FMNIST64/train"

    # resize_custom_mnist_dataset(
    #     dataset_path=CUSTOM_DATASET_PATH,
    #     output_dir=OUTPUT_DIR,
    #     target_size=(64, 64)
    # )

    # 检查数据集是否成功改变大小
    check_image_sizes('./FMNIST64/train')