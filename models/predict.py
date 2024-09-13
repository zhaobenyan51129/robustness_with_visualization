import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.load_model import load_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    imagenet_root = '../imagenet'
    testset = datasets.ImageFolder(imagenet_root + '/val', transform=data_transform)
    dataloader = DataLoader(testset, batch_size=256, num_workers=8)
    model = load_model('vgg16') # 'vit_b_16','resnet50'
    model = model.to(device)

    # 使用 DataParallel 进行多 GPU 预测
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # 计算模型在整个数据集上的准确率
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()