import numpy as np
import torch
import pandas as pd
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from visualization.grad_cam import GradCAM, show_cam_on_image
from visualization.reshape_tranform import ReshapeTransform
from tools.get_classes import get_classes_with_index
from tools.show_images import show_grad, show_images, plot_distribution
from models.load_model import load_model
from data_preprocessor.normalize import apply_normalization
from algorithms.single_step_wrapper import *

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
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

class OneStepAttack:
    def __init__(self, model_str, data_path, root):
        self.model_str = model_str
        self.model = load_model(model_str)
        self.data_path = data_path
        self.root = root
        self.images, self.labels = self.load_data()
        self.original_classes = get_classes_with_index(self.labels)
        self.grad = None
        self.adv_images = None
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
            
    def load_data(self):
        '''加载数据'''
        data = torch.load(self.data_path)
        images, labels = data['images'], data['labels']
        return images, labels

    def compute_grad(self):
        '''计算梯度'''
        self.grad = compute_grad(self.model, self.images, self.labels)
        print(f'grad shape: {self.grad.shape}')

    def show_images_grad(self):
        '''展示图片和梯度以及原始图片的grad-CAM结果'''
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        show_images(self.images, titles=self.original_classes, output_path=self.root, save_name='original.png')
        show_grad(self.grad, output_path=self.root, save_name='grad.png')
        plot_distribution(self.grad, output_path=self.root, save_name='grad_distribution.png')
        self.show_grad_cam(self.images, self.root, 'grad_cam_ori.png')
    
    def show_grad_cam(self, input, output_path, save_name, **kwargs):
        '''计算grad-CAM并展示结果
        Args:
            input: 输入图片
            save_name: 保存的图片名
        '''
        titles = kwargs.get('titles', None)
        main_title = kwargs.get('main_title', None)
        grayscale_cam, vis = run_grad_cam(self.model, input, self.labels, self.target_layers, self.reshape_transform, self.use_cuda)
        show_images(vis, titles=titles, output_path=output_path, save_name=save_name, main_title=main_title)

    def attack(self, algo='fgsm', eta_list=np.arange(0, 0.1, 0.01), mask_mode=None, **kwargs):
        show = False # 是否展示图片
        
        para = kwargs.get(mask_mode, None)
        if para is not None:
            save_path = os.path.join(self.root, algo, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, algo, mask_mode)
            
        if os.path.exists(save_path):
            show = False
        
        mask, num_attacked = grad_mask(self.grad, mode=mask_mode, **kwargs)
        print(f'algo: {algo}, mask_mode: {mask_mode}, num_attacked: {num_attacked}')
        
        perturbations = generate_perturbations(algo, eta_list, self.grad)
        perturbations = [perturbation * mask for perturbation in perturbations]
        self.adv_images = generate_adv_images(self.images, perturbations)

        if show:
            save_path_image = os.path.join(save_path, 'adv_images')
            save_path_cam = os.path.join(save_path, 'grad_cam')
            save_path_perb = os.path.join(save_path, 'perturbation')
            make_dir(save_path_image)
            make_dir(save_path_cam)
            make_dir(save_path_perb)
        
        success_rate_dict = {}
        for i, eta in enumerate(eta_list):
            pred = self.model(apply_normalization(self.adv_images[i])).argmax(dim=1)
            pred_classes = get_classes_with_index(pred)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[eta] = success_rate
            print(f'eta: {eta}, success_rate: {success_rate:.2f}')
            
            if show:
                titles = [f'{original}/{pred}' if original != pred else original for original, pred in zip(self.original_classes, pred_classes)]
                main_title = f'{algo}, eta: {eta}, success_rato: {success_rate:.2f}'
            
                show_images(self.adv_images[i], output_path=save_path_image, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
            
                self.show_grad_cam(self.adv_images[i], output_path=save_path_cam, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
            
                perturbation = perturbations[i].permute(0, 2, 3, 1).detach().cpu().numpy()
                show_images(perturbation, output_path=save_path_perb, titles=titles, save_name=f'{round(eta,2)}.png', main_title=main_title)
                plot_distribution(perturbation, output_path=save_path_perb, save_name=f'distribution_{round(eta,2)}.png')
            
        return num_attacked, success_rate_dict

if __name__ == '__main__':
  
    results = pd.DataFrame(columns=['model','algo', 'mask_mode', 'parameter', 'eta', 'num_attacked', 'success_rate'])
    # 算法列表
    algo_list = ['fgsm', 'fgm', 'gaussian_noise']
    eta_list = np.arange(0.01, 0.31, 0.01)
    
    # algo_list = ['fgsm']
    # eta_list = [0.1]

    # mask_mode列表和对应的参数范围
    mask_modes = {
        'topk': range(100, 20001, 100),
        'topr': np.arange(0.005, 0.5, 0.005),
        # 'topr': [0.1],
        # 'topk': [1000],
        'positive': [None],
        'negative': [None],
        'all': [None],
        'randomk': range(100, 20001, 100),
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    single_root = './data/one_step_attack'
    for model_str in model_list:
        root = os.path.join(single_root, model_str)
        make_dir(root)
        
        image_path = './data/images_new_100.pth'
        attacker = OneStepAttack(model_str, image_path, root)
        attacker.compute_grad()
        # attacker.show_images_grad()
        
        for algo in algo_list:
            for mask_mode, parameters in mask_modes.items():
                for parameter in parameters:
                    # 调用attack方法
                    if parameter is None:
                        num_attacked, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode)
                    else:
                        num_attacked, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode, **{mask_mode: parameter})
                    
                    # 遍历返回的字典中的每一个eta和对应的成功率
                    for eta, success_rate in success_rate_dict.items():
                        # 将结果保存到DataFrame中
                        new_row = pd.DataFrame({
                                    'model': [model_str],
                                    'algo': [algo],
                                    'mask_mode': [mask_mode],
                                    'parameter': [parameter],
                                    'eta': [eta],
                                    'num_attacked': [num_attacked],
                                    'success_rate': [success_rate]
                                })
                        if results.empty:
                            results = new_row
                        else:
                            results = pd.concat([results, new_row], ignore_index=True)
    results.to_excel(os.path.join(single_root, 'result_one_step.xlsx'), index=False)
    # 释放显存
    torch.cuda.empty_cache()