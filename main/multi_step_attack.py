'''多步法'''
import torch
import torch.nn as nn
import sys
import pandas as pd
from tqdm import tqdm
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from visualization.reshape_tranform import ReshapeTransform
from tools.get_classes import get_classes_with_index
from models.load_model import load_model
from data_preprocessor.normalize import apply_normalization
from visualization.grad_cam import GradCAM, show_cam_on_image
from algorithms.single_step_wrapper import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(data_path):
    '''加载数据'''
    data = torch.load(data_path)
    images, labels = data['images'], data['labels']
    return images, labels

def run_grad_cam(model, images, labels, target_layers, reshape_transform, use_cuda):
    '''运行Grad-CAM算法,并返回可视化结果和预测的类别
    Args:
        model: 模型
        images: 图片
        labels: 标签
        target_layers: 目标层
        reshape_transform: 重塑变换
        use_cuda: 是否使用cuda    
    '''
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=use_cuda)
    _, grayscale_cam = cam(images, target_category=labels)
    img = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return grayscale_cam, vis

class MultiStepAttack:
    def __init__(self, model_str, images, labels, root):
        self.model_str = model_str
        self.model = load_model(model_str)
        self.root = root
        self.images = images
        self.labels = labels
        self.original_classes = get_classes_with_index(self.labels)
        if model_str == 'vit_b_16':
            self.target_layers = [self.model.blocks[-1].norm1]
            self.reshape_transform = ReshapeTransform(self.model)
        elif model_str == 'resnet50':
            self.target_layers = [self.model.layer4[-1].conv3]
            self.reshape_transform = None
        elif model_str == 'vgg16':
            self.target_layers = [self.model.features[-1]]
            self.reshape_transform = None
        else:
            raise Exception('model_str error!')
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
    
    def compute_loss_function(self, output, y, loss_mode):
        if loss_mode == 'CrossEntropy':
            loss = nn.CrossEntropyLoss()(output, y)
        elif loss_mode == 'logit':
            loss = get_loss(output, y)
        elif loss_mode == 'softmax':
            output = nn.Softmax(dim=1)(output)
            loss = get_loss(output, y)
        return loss
    
    def i_fgsm(self, alpha, eta, num_steps, loss_mode = 'CrossEntropy', mask_mode ='all', **kwargs):
        """ Construct I-FGSM adversarial examples on the examples X
        
        Args:
            alpha: 扰动的步长
            eta: 扰动阈值  
            num_steps: 迭代次数
            loss_mode: 损失函数的模式，str, 可选：'CrossEntropy'，'logit', 'softmax', default: 'CrossEntropy'  
            mask_mode: 计算需要保留梯度的pixel，同单步法，str, 可选：'all', 'positive', 'negative', 'topr', 'randomr', 'cam_topr',default: 'all'
        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
        """
        delta = torch.zeros_like(self.images, requires_grad=True)
        success_rate_dict = {}
        for t in tqdm(range(num_steps), desc="Processing steps"):
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate
            
            loss = self.compute_loss_function(output, self.labels, loss_mode)
            loss.backward()
            grad = delta.grad.detach().clone()
            if mask_mode == 'cam_topr':
                grayscale_cam, _ = run_grad_cam(self.model, self.images, self.labels, self.target_layers, self.reshape_transform, self.use_cuda)
                mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
            else:
                mask, _ = grad_mask(grad, mode=mask_mode, **kwargs)
            delta.data = delta.data + alpha * mask * grad.sign()
            delta.data = torch.clamp(delta.data, -eta, eta)
            delta.grad.zero_()  
            
        return delta, success_rate_dict
    
    def i_gauss(self, alpha, eta, num_steps, loss_mode = 'CrossEntropy', mask_mode ='all', **kwargs):
        """ Construct I-Gauss adversarial examples on the examples X
        
        Args:
            alpha: 扰动的步长
            eta: 扰动阈值  
            num_steps: 迭代次数
            loss_mode: 损失函数的模式，str, 可选：'CrossEntropy'，'logit', 'softmax', default: 'CrossEntropy'  
            mask_mode: 计算需要保留梯度的pixel，同单步法，str, 可选：'all', 'positive', 'negative', 'topr', 'randomr', 'cam_topr',default: 'all'
        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
        """
        delta = torch.zeros_like(self.images, requires_grad=True)
        success_rate_dict = {}
        for t in tqdm(range(num_steps), desc="Processing steps"):
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate
            
            if mask_mode == 'cam_topr':
                grayscale_cam, _ = run_grad_cam(self.model, self.images, self.labels, self.target_layers, self.reshape_transform, self.use_cuda)
                mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
            else:
                loss = self.compute_loss_function(output, self.labels, loss_mode)
                loss.backward()
                grad = delta.grad.detach().clone()
                mask, _ = grad_mask(grad, mode=mask_mode, **kwargs)
            delta.data = delta + alpha * mask * torch.randn_like(grad) # Gaussian noise
            delta.data = torch.clamp(delta.data, -eta, eta)
            delta.grad.zero_()  
            
        return delta, success_rate_dict

if __name__ == "__main__":
    model_str = 'vit_b_16'
    image_path = f'./data/imges_classified/14.pth'
    images, labels = load_data(image_path)
    images = images.to(device)
    labels = labels.to(device)
    root = './data/multi_step_attack_test'
    make_dir(root)
    attacker = MultiStepAttack(model_str, images, labels, root)
    
    delta, success_rate_dict = attacker.i_fgsm(alpha=0.02, eta=0.1, num_steps=10, mask_mode='all')
    torch.save(delta, './data/multi_step_attack_test/delta_ifgsm.pth')

    success_rate_df = pd.DataFrame(list(success_rate_dict.items()), columns=['Step', 'Success Rate'])
    success_rate_df.to_excel(os.path.join(root, 'success_rate_dict_ifgsm.xlsx'), index=False)
    print(delta.shape)
    print('ifgsm done!')
    
    delta, success_rate_dict = attacker.i_gauss(alpha=1, eta=0.1, num_steps=10, mask_mode='all')
    torch.save(delta, './data/multi_step_attack_test/delta_igauss.pth')
    success_rate_df = pd.DataFrame(list(success_rate_dict.items()), columns=['Step', 'Success Rate'])
    success_rate_df.to_excel(os.path.join(root, 'success_rate_dict_igauss.xlsx'), index=False)
    print(delta.shape)
    print('igauss done!')







