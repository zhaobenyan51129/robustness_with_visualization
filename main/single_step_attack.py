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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
    def __init__(self, model_str, images, labels, root):
        self.model_str = model_str
        self.model = load_model(model_str)
        self.root = root
        self.images = images
        self.labels = labels
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
            attack_ratio_per_channel: 每个通道被攻击的比例
            l1_norm: 被攻击像素的L1范数
            l2_norm_squre: 被攻击像素的L2范数平方
            original_loss: 原始损失
            attack_after_loss_dict: 攻击后的损失字典
        '''
        
        # 获取掩码参数并构建保存路径
        para = kwargs.get(mask_mode, None)
        if para is not None:
            save_path = os.path.join(self.root, algo, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, algo, mask_mode)
        
        # 计算原始损失
        self.model.zero_grad()
        delta = torch.zeros_like(self.images, requires_grad=True).to(device)
        output = self.model(self.images + delta)
        loss_fn = nn.CrossEntropyLoss()
        ori_loss = loss_fn(output, self.labels)
        ori_loss.backward()
        original_loss = - round(ori_loss.item(), 3)
        self.grad = delta.grad.detach().clone()
        # 根据mask_mode选择掩码生成方式
        if mask_mode in ['cam_lowr', 'cam_topr']:
            self.grayscale_cam, self.vis = run_grad_cam(
                self.model, self.images, self.labels, self.target_layers, 
                self.reshape_transform, self.use_cuda
            )
            mask, pixel_attacked = cam_mask(self.grayscale_cam, mode=mask_mode, **kwargs)
        else:
            mask, pixel_attacked = grad_mask(self.grad, mode=mask_mode, **kwargs)
        
        # 生成扰动并应用掩码
        perturbations = generate_perturbations(algo, eta_list, self.grad)
        perturbations = [perturbation * mask for perturbation in perturbations]
        self.adv_images = generate_adv_images(self.images, perturbations)
        
        # 计算被攻击像素的L1范数和L2范数平方
        masked_grad = self.grad * mask
        l1_norm = round(masked_grad.abs().sum().cpu().item(), 4)
        l2_norm_squre = round((masked_grad ** 2).sum().cpu().item(), 6)
        
        # 计算每个通道被攻击的像素比例
        attacked_pixels_per_channel = mask.sum(dim=(0, 2, 3))
        total_pixels_per_channel = mask.size(0) * mask.size(2) * mask.size(3)
        attack_ratio_per_channel = attacked_pixels_per_channel / total_pixels_per_channel
        attack_ratio_per_channel = [round(x, 3) for x in attack_ratio_per_channel.tolist()]
        
        success_rate_dict = {}
        attack_after_loss_dict = {}  # 用于存储每个扰动下的攻击后损失
        
        for i, eta in enumerate(eta_list):
            # 获取对应的对抗样本
            adv_image = self.adv_images[i].detach().clone().requires_grad_(True)
            
            # 计算对抗样本的输出和损失
            adv_output = self.model(adv_image)
            adv_loss = loss_fn(adv_output, self.labels)
            attack_after_loss = adv_loss.item()  # 获取标量损失值
            attack_after_loss_dict[eta] = - round(attack_after_loss, 3)  # 保存攻击后的损失
            
            # 计算成功率
            pred = adv_output.argmax(dim=1)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[eta] = success_rate
        
        return pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, attack_after_loss_dict


def parameter_sample():
    algo_list = ['fgsm']
    eta_list = [0.1]
    
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        'all': [None],
        'topr': [0.3], 
        # 'lowr': [0.3],
        # 'randomr': [0.3],
        # 'channel_randomr': [0.3], # 不能超过1/3
        # 'cam_topr': [0.3],  
        'cam_lowr': [0.3], 
    }
    # model_list = ['vit_b_16', 'resnet50', 'vgg16']
    model_list = ['vit_b_16']
    data_root = './data_stage2/one_step_sample_100_0919_test_loss'
    dataset_file = './data_stage2/images_100_0911.pth'
    save_result_file = 'one_step_sample100_0919_test_loss.xlsx'
    return algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file
    
def parameter_total():
    algo_list = ['fgsm', 'gaussian_noise']
    eta_list = np.arange(0.005, 0.105, 0.005)
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': np.arange(0.01, 1, 0.01),
        'lowr': np.arange(0.01, 1, 0.01),
        'randomr': np.arange(0.01, 1, 0.01),
        # 'channel_randomr': np.arange(0.05, 0.3, 0.05),
        'cam_topr': np.arange(0.01, 1, 0.01),
        'cam_lowr': np.arange(0.01, 1, 0.01),
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    size = 1000
    if size == 1000:
        data_root = './data_stage2/one_step_attack_total1000_0914'
        dataset_file = './data_stage2/images_1000_0914.pth'
        save_result_file = 'one_step_attack_total1000_0919_loss.xlsx'
    else:
        data_root = './data_stage2/one_step_attack_total100_0919'
        dataset_file = './data_stage2/images_100_0911.pth'
        save_result_file = 'one_step_attack_total100_0919_normed.xlsx'
    return algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file

def main():
    results = pd.DataFrame(columns=['model', 'algo', 'mask_mode', 'parameter', 'eta', 'pixel_attacked', 'attack_ratio_per_channel', 'l1_norm', 'l2_norm','success_rate', 'original_loss', 'attack_after_loss_dict', 'run_time', 'batch_idx', 'batch_pictures'])
    # algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file = parameter_sample()
    algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file = parameter_total()
    print(f'data_root is {data_root}')

    dataset = CustomDataset(dataset_file)
    # dataset = torch.utils.data.Subset(dataset, range(16)) # 验证阶段，取16张
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        root = os.path.join(data_root, model_str)
        make_dir(root)

        # 遍历 DataLoader
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1): # 1表示从1开始计数
            batch_pictures = images.size(0)
            attacker = OneStepAttack(model_str, images, labels, root)
            
            for algo in algo_list:
                for mask_mode, parameters in mask_modes.items():
                    if algo == 'gaussian_noise' and mask_mode not in ('all', 'positive', 'negative'):
                        continue
                    for parameter in tqdm(parameters, desc="Parameters", leave=False):
                        # 记录运行时间
                        start_time = time.time()
                        if parameter is None:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, attack_after_loss_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode)  
                        else:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, attack_after_loss_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode, **{mask_mode: parameter}) 
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
                                        'l1_norm': [l1_norm],
                                        'l2_norm': [l2_norm_squre],
                                        'success_rate': [success_rate],
                                        'original_loss': [original_loss],
                                        'attack_loss': [attack_after_loss_dict[eta]],
                                        'run_time': [run_time],
                                        'batch_idx': [batch_idx],
                                        'batch_pictures': [batch_pictures]
                                    })
                            if results.empty:
                                results = new_row
                            else:
                                results = pd.concat([results, new_row], ignore_index=True)
                        # print(f'{model_str}, {algo}, {mask_mode}, {parameter} is finished!')
            torch.cuda.empty_cache()
    results.to_excel(os.path.join(data_root, save_result_file), index=False)
    
if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
    
    