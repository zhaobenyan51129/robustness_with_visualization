'''从imagenet数据集中筛选用于实验的数据，筛选出来的图片要满足所有模型都预测正确'''
import torch
import torchvision
import random
from normalize import apply_normalization


class SelectImageNet:
    def __init__(self, models, data_root='../imagenet', load_images_num = 100, image_size=224):
        '''初始化

        Args:
            models: 模型列表，如 [model1, model2, model3]
            data_root: imagenet数据集根目录, 默认'../imagenet'
            load_images_num: 载入图片数量, 默认100
            image_size: 图片大小, 默认224'''
        self.models = models
        self.data_root = data_root
        self.load_images_num = load_images_num
        self.image_size = image_size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()])
        self.testset = torchvision.datasets.ImageFolder(data_root + '/val', self.transform)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def get_preds(self, model, X):
        '''获取指定模型的预测结果

        Args:
            model: 模型， 
            X: 图片数据， [b, 224, 224, 3]
        '''
        X = X.to(self.device)

        max_value_and_idx = model(apply_normalization(X)).max(dim=1) # 注意送入模型前执行标准的normalize流程
        return max_value_and_idx[1], max_value_and_idx[0] # 获得预测的label和对应概率
    
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

def main():
    '''测试'''
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from models.load_model import load_model
    model1 = load_model('resnet50')
    model2 = load_model('vit_b_32')
    models = [model1, model2]
    data_root = './imagenet'
    select = SelectImageNet(models, data_root=data_root, load_images_num=9, image_size=224)
    select.creat_image_file('select_images.pth')
    
if __name__ == '__main__':
    main()