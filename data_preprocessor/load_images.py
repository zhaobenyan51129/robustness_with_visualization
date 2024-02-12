'''从select_images.py生成的二进制文件中加载图片'''
import torch

def load_images(file_path):
    '''从select_images.py生成的二进制文件中加载图片

    Args:
        file_path: 二进制文件路径

    Returns:
        images: 图片数据, [b, 3, 224, 224], 数据类型为torch.float32
        labels: 标签, [b,], 数据类型为torch.int64'''
    data = torch.load(file_path)
    return data['images'], data['labels']