'''从imagenet数据集中筛选用于实验的数据，筛选出来的图片要满足所有模型都预测正确'''
import torch
# import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import random
import json
from tqdm import tqdm
import os
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.load_model import load_model


class SelectImageNet:
    def __init__(self, models, imagenet_root='imagenet', load_images_num = 100, image_size=224):
        '''初始化

        Args:
            models: 模型列表，如 [model1, model2, model3]
            imagenet_root: imagenet数据集根目录, 默认'imagenet'
            load_images_num: 载入图片数量, 默认100
            image_size: 图片大小, 默认224
        '''
        self.models = models
        self.imagenet_root = imagenet_root
        self.load_images_num = load_images_num
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.testset = datasets.ImageFolder(imagenet_root + '/val', self.transform)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device {device}')
        self.device = device

    def get_preds(self, model, X):
        '''获取指定模型的预测结果

        Args:
            model: 模型， 
            X: 图片数据， [b, 224, 224, 3]
        '''
        X = X.to(self.device)
        max_value_and_idx = model(X).max(dim=1) 
        return max_value_and_idx[1], max_value_and_idx[0] # 获得预测的label和对应概率
    
    def get_classes_acc(self, save_path):
        '''计算模型在验证集的每个类别上的准确率，保存为json文件
        Args:
            save_path: 保存的文件路径
        '''
        dataloader = DataLoader(self.testset, batch_size=64, shuffle=True) # 长度是有多少个batch
        class_correct = list(0. for i in range(len(self.testset.classes)))
        class_total = list(0. for i in range(len(self.testset.classes)))
        model = self.models[0].to(self.device) 
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                outputs = model(images.to(self.device))
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        class_acc = {self.testset.classes[i]: class_correct[i] / class_total[i] for i in range(len(self.testset.classes))}
        with open(save_path, 'w') as f:
            json.dump(class_acc, f)
        
    def creat_image_file(self, save_path):
        '''保存所有模型均预测正确的图片和标签

        Args:
            save_path: 保存的文件路径'''
        images = torch.zeros(self.load_images_num, 3, self.image_size, self.image_size).to(self.device) # [100, 3, 224, 224]
        labels = torch.zeros(self.load_images_num).long().to(self.device) # [100,][0,0,...,0]
        preds = labels + 1 # [100, ][1,1,...,1]
        preds = preds.to(self.device)
        while preds.ne(labels).sum() > 0: # 如果预测值和label不相等的个数不为0 ne:不相等返回1
            idx = torch.arange(0, images.size(0)).long().to(self.device)[preds.ne(labels)] # 计算没预测对的的位置， .long-> torch.int64
            for i in list(idx):
                images[i], labels[i] = self.testset[random.randint(0, len(self.testset) - 1)] # 0~49999
            pred_labels = []
            for model in self.models:
                pred_label, _ = self.get_preds(model, images[idx])
                pred_labels.append(pred_label)
                if len(set(pred_labels)) > 1: # 如果所有模型预测得不一样，则跳出循环，重新选图片
                    break
                else: # 如果所有模型都预测的一样，则保存
                    preds[idx] = pred_labels[0]
        torch.save({'images': images, 'labels': labels}, save_path)
        
    def load_select_images(self, num_images_per_class, class_list, save_path):
        '''加载指定类别的图片和标签,要求所有模型都预测正确
        Args:
            num_images_per_class: 每个类别的图片数量
            class_list: 类别列表
            save_path: 保存的文件路径
        '''
        saved_images_count = {class_id: 0 for class_id in class_list}
        saved_images = {class_id: [] for class_id in class_list}
        dataloader = DataLoader(self.testset, batch_size=64, shuffle=True)

        # 遍历数据集并添加并行化处理
      
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(self.device)
                # labels = labels.to(device)
                model = self.models[0]
                if torch.cuda.device_count() > 1:
                    # print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)
                # model = self.models[0].to(self.device)
                model.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                correct = (predicted == labels).tolist()
                
                for i, is_correct in enumerate(correct):   
                    class_id = str(labels[i].item())
                    if is_correct and class_id in class_list and saved_images_count[class_id] < num_images_per_class:
                        saved_images[class_id].append(images[i])
                        saved_images_count[class_id] += 1
                    if all(count >= num_images_per_class for count in saved_images_count.values()):
                        break
                    
        for class_id, images in saved_images.items():
            print(f'Class {class_id}: {len(images)} images')
            # 将图片和标签转化为tensor并保存
            images = torch.stack(images)
            labels = torch.tensor([int(class_id)] * len(images))
            
            torch.save({'images': images, 'labels': labels}, os.path.join(save_path, f'{class_id}.pth'))

def main():
    '''测试'''
    # model1 = load_model('resnet50')
    model2 = load_model('vit_b_16')
    # models = [model1, model2]
    models = [model2]
    
    imagenet_root = '../imagenet'       
    data_dir = './data'
    select = SelectImageNet(models, imagenet_root=imagenet_root, load_images_num=100, image_size=224)
    select.creat_image_file(os.path.join(data_dir, 'images_100.pth'))
    # select.get_classes_acc(os.path.join(data_dir, 'class_acc.json'))

def main_generate_data():
    model = load_model('vit_b_16')
    models = [model]
    # class_list = ['335']
    class_list = ['14', '900', '335', '75', '870', '50', '159', '793', '542', '675', '664']  
    imagenet_root = '../imagenet'       
    data_dir = './data'
    select = SelectImageNet(models, imagenet_root=imagenet_root, load_images_num=100, image_size=224)
    select.load_select_images(100, class_list, data_dir)
    
     
if __name__ == '__main__':
    # main()
    main_generate_data()