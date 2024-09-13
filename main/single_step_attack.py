import time
import numpy as np
import torch
import pandas as pd
import sys
import os
from tqdm import tqdm 

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from visualization.grad_cam import GradCAM, show_cam_on_image
from visualization.reshape_tranform import ReshapeTransform
from tools.get_classes import get_classes_with_index
from tools.show_images import show_grad, show_images, plot_distribution, show_mask
from models.load_model import load_model
from data_preprocessor.load_images import CustomDataset
from torch.utils.data import DataLoader
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
    img = images.permute(0, 2, 3, 1).detach().cpu().numpy() # [batch, 224, 224, 3]
    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return grayscale_cam, vis

class OneStepAttack:
    def __init__(self, model_str, images, labels, root, show=False):
        self.model_str = model_str
        self.model = load_model(model_str)
        self.root = root
        self.images = images
        self.labels = labels
        self.show = show
        self.original_classes = get_classes_with_index(self.labels)
        self.grad = None
        self.adv_images = None
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        if model_str == 'vit_b_16':
            self.target_layers = [self.model.blocks[-1].norm1]
            self.reshape_transform = ReshapeTransform(self.model)
        elif model_str == 'resnet50':
            self.target_layers = [self.model.layer4[-1].conv3]
            # self.target_layers = [self.model.layer4[-1]]
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

    def compute_grad_and_cam(self):
        '''计算梯度'''
        self.grad = compute_grad(self.model, self.images, self.labels)
        self.grayscale_cam, self.vis = run_grad_cam(self.model, self.images, self.labels, self.target_layers, self.reshape_transform, self.use_cuda)

    def show_images_grad(self):
        '''展示图片和梯度以及原始图片的grad-CAM结果'''
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        show_images(self.images, titles=self.original_classes, output_path=self.root, save_name='original.png')
        show_grad(self.grad, output_path=self.root, save_name='grad.png')
        plot_distribution(self.grad, output_path=self.root, save_name='grad_distribution.png')
        show_images(self.vis, titles=self.original_classes, output_path=self.root, save_name='grad_cam_ori.png')

    def attack(self, algo='fgsm', eta_list=np.arange(0, 0.1, 0.01), mask_mode=None, **kwargs):
        '''对模型进行单步法攻击
        Args:
            algo: 攻击算法
            eta_list: 扰动的范围
            mask_mode: 选择攻击像素的方式
            kwargs: 掩码模式的参数
        return:
            pixel_attacked: 攻击的像素
            success_rate_dict: 成功率字典
        '''
        para = kwargs.get(mask_mode, None)
        if para is not None:
            save_path = os.path.join(self.root, algo, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, algo, mask_mode)
            
        if os.path.exists(save_path):
            self.show = False
            
        if mask_mode in ['cam_lowr', 'cam_topr']:
            mask, pixel_attacked = cam_mask(self.grayscale_cam, mode=mask_mode, **kwargs)
        else:
            mask, pixel_attacked = grad_mask(self.grad, mode=mask_mode, **kwargs)
        
        perturbations = generate_perturbations(algo, eta_list, self.grad)
        perturbations = [perturbation * mask for perturbation in perturbations]
        self.adv_images = generate_adv_images(self.images, perturbations)
        
        # mask形状为[batch, 3, 224, 224]，计算每个通道的像素被攻击的比例
        attacked_pixels_per_channel = mask.sum(dim=(0, 2, 3))
        total_pixels_per_channel = mask.size(0) * mask.size(2) * mask.size(3)
        attack_ratio_per_channel = attacked_pixels_per_channel / total_pixels_per_channel
        attack_ratio_per_channel = [round(x, 3) for x in attack_ratio_per_channel.tolist()]

        if self.show:
            save_path_image = os.path.join(save_path, 'adv_images')
            save_path_cam = os.path.join(save_path, 'grad_cam')
            save_path_perb = os.path.join(save_path, 'perturbation')
            make_dir(save_path_image)
            make_dir(save_path_cam)
            make_dir(save_path_perb)
            show_mask(mask, output_path=save_path_perb, save_name='mask.png')
            
        success_rate_dict = {}
        for i, eta in enumerate(eta_list):
            pred = self.model(self.adv_images[i]).argmax(dim=1)
            pred_classes = get_classes_with_index(pred)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[eta] = success_rate
            
            if self.show:
                titles = [f'{original}/{pred}' if original != pred else original for original, pred in zip(self.original_classes, pred_classes)]
                main_title = f'{algo}, eta: {eta}, success_rate: {success_rate:.2f}'
            
                show_images(self.adv_images[i], output_path=save_path_image, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
                
                _, vis = run_grad_cam(self.model, self.adv_images[i], pred, self.target_layers, self.reshape_transform, self.use_cuda)
                show_images(vis, output_path=save_path_cam, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
            
                perturbation = perturbations[i].permute(0, 2, 3, 1).detach().cpu().numpy()
                show_images(perturbation, output_path=save_path_perb, titles=titles, save_name=f'{round(eta,2)}.png', main_title=main_title)
                # plot_distribution(perturbation, output_path=save_path_perb, save_name=f'distribution_{round(eta,2)}.png')
                
        return pixel_attacked, success_rate_dict, attack_ratio_per_channel

def parameter_sample():
    algo_list = ['fgsm', 'gaussian_noise']
    eta_list = [0.1]
    
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        # 'all': [None],
        # 'topr': [0.3], 
        # 'lowr': [0.3],
        # 'randomr': [0.3],
        # 'channel_randomr': [0.3], # 不能超过1/3
        'cam_topr': [0.5],  
        'cam_lowr': [0.5]
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    # single_root = './data/one_step_attack_sample_900'
    single_root = './data_stage2/one_step_sample_100_0912_test'
    return algo_list, eta_list, mask_modes, model_list, single_root
    
def parameter_total():
    algo_list = ['fgsm', 'gaussian_noise'] # 删除了'fgm'
    eta_list = np.arange(0.01, 0.21, 0.01)
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': np.arange(0.1, 1, 0.1),
        'lowr': np.arange(0.1, 1, 0.1),
        'randomr': np.arange(0.1, 1, 0.1),
        # 'channel_randomr': np.arange(0.05, 0.3, 0.05),
        'cam_topr': np.arange(0.1, 1, 0.1),
        'cam_lowr': np.arange(0.1, 1, 0.1),
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    single_root = './data_stage2/one_step_attack_total_0913'
    return algo_list, eta_list, mask_modes, model_list, single_root

def main():
    results = pd.DataFrame(columns=['model', 'algo', 'mask_mode', 'parameter', 'eta', 'pixel_attacked', 'attack_ratio_per_channel', 'success_rate', 'run_time', 'batch'])
    # algo_list, eta_list, mask_modes, model_list, single_root = parameter_sample()
    algo_list, eta_list, mask_modes, model_list, single_root = parameter_total()
    print(f'data_root is {single_root}')

    dataset = CustomDataset('./data/images_100_0911.pth')
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        root = os.path.join(single_root, model_str)
        make_dir(root)

        # 遍历 DataLoader
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1): # 1表示从1开始计数
            attacker = OneStepAttack(model_str, images, labels, root, show)
            attacker.compute_grad_and_cam()
            # attacker.show_images_grad()
            
            for algo in tqdm(algo_list, desc="Algorithms", leave=False):
                for mask_mode, parameters in tqdm(mask_modes.items(), desc="Mask Modes", leave=False):
                    for parameter in tqdm(parameters, desc="Parameters", leave=False):
                        # 记录运行时间
                        start_time = time.time()
                        if parameter is None:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode)  
                        else:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode, **{mask_mode: parameter}) 
                        end_time = time.time()
                        run_time = round(end_time - start_time, 3)
                    
                        # 遍历返回的字典中的每一个eta和对应的成功率
                        for eta, success_rate in success_rate_dict.items():
                            # 将结果保存到DataFrame中
                            new_row = pd.DataFrame({
                                        'model': [model_str],
                                        'algo': [algo],
                                        'mask_mode': [mask_mode],
                                        'parameter': [parameter],
                                        'eta': [eta],
                                        'pixel_attacked': [pixel_attacked],
                                        'attack_ratio_per_channel': [attack_ratio_per_channel],
                                        'success_rate': [success_rate],
                                        'run_time': [run_time],
                                        'batch': [batch_idx]
                                    })
                            if results.empty:
                                results = new_row
                            else:
                                results = pd.concat([results, new_row], ignore_index=True)
                        # print(f'{model_str}, {algo}, {mask_mode}, {parameter} is finished!')
            torch.cuda.empty_cache()
    results.to_excel(os.path.join(single_root, 'result_one_step_sample100_0912.xlsx'), index=False)
    
if __name__ == '__main__':
    show = False
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
    
    