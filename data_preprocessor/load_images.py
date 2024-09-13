'''从select_images.py生成的二进制文件中加载图片'''
import torch
from torch.utils.data import Dataset, DataLoader

def load_images(file_path):
    '''从select_images.py生成的二进制文件中加载图片

    Args:
        file_path: 二进制文件路径

    Returns:
        images: 图片数据, [b, 3, 224, 224], 数据类型为torch.float32
        labels: 标签, [b,], 数据类型为torch.int64'''
    data = torch.load(file_path)
    return data['images'], data['labels']

class CustomDataset(Dataset):
    def __init__(self, file_path):
        # 加载保存的数据
        data = torch.load(file_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

if __name__ == '__main__':
    # 使用自定义的 Dataset 类
    dataset = CustomDataset('./data/images_100_0911.pth')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 示例：遍历 DataLoader
    for images, labels in dataloader:
        print(images.shape, labels.shape)