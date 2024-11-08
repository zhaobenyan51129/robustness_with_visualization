'''从imagenet数据集中筛选用于实验的数据，筛选出来的图片要满足所有模型都预测正确'''
import torch
# import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import json
from tqdm import tqdm
import os
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.load_model import load_model


class SelectImageNet:
    def __init__(self, models, imagenet_root='imagenet', load_images_num = 100, image_size=224):
        '''初始化 SelectImageNet 类筛选指定模型全部预测正确的图片

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
        '''获取指定模型的预测结果,按batch计算

        Args:
            model: 模型， 
            X: 图片数据， [b, 224, 224, 3]
        '''
        model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            max_value_and_idx = model(X).max(dim=1) 
            return max_value_and_idx[1], max_value_and_idx[0]
        
    def creat_image_file(self, save_path):
        '''保存所有模型均预测正确的图片和标签

        Args:
            save_path: 保存的文件路径'''
        images = torch.zeros(self.load_images_num, 3, self.image_size, self.image_size).to(self.device) # [100, 3, 224, 224]
        labels = torch.zeros(self.load_images_num).long().to(self.device) # [100,][0,0,...,0]
        preds = labels + 1 # [100, ][1,1,...,1]
        preds = preds.to(self.device)
        while preds.ne(labels).sum() > 0: # 如果预测值和label不相等的个数不为0 ne:不相等返回1
            idx = torch.arange(0, images.size(0)).long().to(self.device)[preds.ne(labels)] # 计算没预测对的的位置
            for i in list(idx):
                images[i], labels[i] = self.testset[random.randint(0, len(self.testset) - 1)] # 0~49999
            pred_labels = []
            for model in self.models:
                pred_label, _ = self.get_preds(model, images[idx])
                pred_labels.append(pred_label)
            # 如果所有模型都预测正确，则将preds中对应位置设置为预测值
            for i, num in enumerate(idx):
                if all([pred_label[i] == labels[num] for pred_label in pred_labels]):
                    preds[num] = labels[num]
        torch.save({'images': images, 'labels': labels}, save_path)

class SelectClassImageNet():
    '''从imagenet数据集中筛选指定类别的图片'''
    def __init__(self, imagenet_root, model_str, batch_size):
        self.imagenet_root = imagenet_root
        self.model_str = model_str
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = load_model(model_str)
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        
        # 设置数据预处理
        if model_str == 'vit_b_16':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # 加载数据集
        self.testset = datasets.ImageFolder(root=os.path.join(imagenet_root, 'val'), transform=self.transform)
        self.dataloader = DataLoader(self.testset, batch_size=batch_size, shuffle=True, num_workers=8)

    def get_preds(self, model, X):
        '''获取指定模型的预测结果,按batch计算

        Args:
            model: 模型， 
            X: 图片数据， [b, 3, 224, 224]
        '''
        model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = model(X)
            max_value_and_idx = outputs.max(dim=1) 
            return max_value_and_idx[1], max_value_and_idx[0]  

    def get_classes_acc(self, save_json_path):
        '''计算模型在验证集的每个类别上的准确率，保存为json文件
        Args:
            save_path: 保存的文件路径
        '''
        self.save_json_path = save_json_path
        class_correct = list(0. for i in range(len(self.testset.classes)))
        class_total = list(0. for i in range(len(self.testset.classes)))
        model = self.model.to(self.device)  # 使用初始化的模型
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        class_acc = {
            self.testset.classes[i]: class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            for i in range(len(self.testset.classes))
        }
        # 按准确率排序
        sorted_class_acc = dict(sorted(class_acc.items(), key=lambda item: item[1], reverse=True))
        with open(save_json_path, 'w') as f:
            json.dump(sorted_class_acc, f)
    
    def select_class(self, ranks, imagenet_class_index_path='./data/imagenet_class_index.json', chinese_class_index_path = './data/chinese_class_index.json'):
        ''' 筛选出指定排名的类别
        
        Args:
        - ranks: list, 指定的排名
        - imagenet_class_index_path: str, imagenet类别索引文件路径
        - chinese_class_index_path: str, 中文类别索引文件路径
        '''
        with open(imagenet_class_index_path) as f:
            class_index_dict = json.load(f)

        with open(chinese_class_index_path) as f:
            chinese_class_index_dict = json.load(f)
        
        acc = json.load(open(self.save_json_path))
        
        sorted_acc = sorted(acc.items(), key=lambda x: x[1], reverse=True)
        selected_items = {k: v for i, (k, v) in enumerate(sorted_acc) if i+1 in ranks}
        result = {}
        for rank, (class_code, accuracy) in zip(ranks, selected_items.items()):
            for id, (code, name) in class_index_dict.items():
                if code == class_code:
                    chinese_name = chinese_class_index_dict[str(id)]
                    corect_num = int(accuracy * 50)
                    result[id] = [class_code, name, chinese_name, accuracy, corect_num, rank]
                    break
        return result

    def load_select_images(self, num_images_per_class, class_list, save_path):
        '''加载指定类别的图片和标签
        Args:
            num_images_per_class: 每个类别的图片数量
            class_list: 类别列表（字符串形式的类别索引，例如 ['0', '1', '2']）
            save_path: 保存的文件路径
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saved_images_count = {class_id: 0 for class_id in class_list}
        saved_images = {class_id: [] for class_id in class_list}
        # 创建新的 DataLoader，确保数据被打乱
        dataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).cpu().tolist()
                labels = labels.cpu()
                for i, is_correct in enumerate(correct):   
                    class_id = str(labels[i].item())
                    if (is_correct and class_id in class_list and 
                        saved_images_count[class_id] < num_images_per_class):
                        saved_images[class_id].append(images[i].cpu())
                        saved_images_count[class_id] += 1
                        # 检查是否已收集足够的图片
                        if all(count >= num_images_per_class for count in saved_images_count.values()):
                            break
                if all(count >= num_images_per_class for count in saved_images_count.values()):
                    break
                        
        for class_id, images_list in saved_images.items():
            print(f'Class {class_id}: {len(images_list)} images')
            # 将图片和标签转化为tensor并保存
            if images_list:
                images_tensor = torch.stack(images_list).to(self.device)
                labels_tensor = torch.tensor([int(class_id)] * len(images_list)).to(self.device)
                torch.save({'images': images_tensor, 'labels': labels_tensor}, os.path.join(save_path, f'{class_id}.pth'))

def main():
    '''测试, 生成数据集ADV1000 ADV100'''
    model1 = load_model('vit_b_16')
    model2 = load_model('resnet50')
    model3 = load_model('vgg16')
    models = [model1, model2, model3]
    models = [model.module if isinstance(model, nn.DataParallel) else model for model in models]
            
    imagenet_root = '../imagenet'       
    data_dir = './data_stage2'
    select = SelectImageNet(models, imagenet_root=imagenet_root, load_images_num=1000, image_size=224)
    select.creat_image_file(os.path.join(data_dir, 'images_1000_0914.pth'))

def main_class_data():
    model_str = 'vgg16'
    imagenet_root = '../imagenet'
    batch_size = 128
    select = SelectClassImageNet(imagenet_root, model_str, batch_size)
    # 保存类别准确率
    select.get_classes_acc(f'./data_stage3/class_acc_{model_str}.json')
    # 选择指定类别
    # ranks = [1, 2, 3, 4, 5]
    # selected_items = select.select_class(ranks)
    
    # 加载指定类别预测正确的图片
    # num_images_per_class = 50
    # # class_list = ['1', '512', '569', '642', '959', '680', '314', '468', '382', '460', '782']
    # class_list = ['46', '74', '110', '167', '174', '230', '240', '241', '249', '254', '282', '369', '408', '414', '423', '460', '482', '492', '501', '534', '552', '620', '638', '664', '675', '689', '723', '725', '733', '741', '751', '782', '848', '876', '948', '961', '968']
    # save_path = './data_stage3/images_classified'
    # select.load_select_images(num_images_per_class, class_list, save_path)
    
     
if __name__ == '__main__':
    # main()
    main_class_data()