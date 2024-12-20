import numpy as np
import torch
import sys
import os

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from visualization.grad_cam import GradCAM, show_cam_on_image
from visualization.reshape_tranform import ReshapeTransform
from tools.get_classes import get_classes_with_index
from tools.show_images import visualize_masks_overlay, show_images, show_pixel_distribution, visualize_gradients
from models.load_model import load_model
from algorithms.single_step_wrapper import *
from algorithms.LRP.lrp import LRPModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
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
    def __init__(self, model_str, images, labels, root, **kwargs):
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
        
        batch_size = images.size(0)
        self.ncols = kwargs.get('ncols', int(np.ceil(np.sqrt(batch_size))))
        self.nrows = kwargs.get('nrows', int(np.ceil(batch_size / self.ncols)))

    def attack(self, algo='fgsm', eta_list=np.arange(0, 0.1, 0.01), mask_mode=None, show = False, **kwargs):
        '''对模型进行单步法攻击
        Args:
            algo: 攻击算法
            eta_list: 扰动的范围
            mask_mode: 选择攻击像素的方式
            show: 是否画图
            kwargs: 掩码模式的参数
        return:
            pixel_attacked: 攻击的像素
            success_rate_dict: 成功率字典
            attack_ratio_per_channel: 每个通道被攻击的比例
            l1_norm: 被攻击像素的L1范数
            l2_norm_squre: 被攻击像素的L2范数平方
            original_loss: 原始损失
            loss_dict_attacked: 攻击后的损失字典
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
        original_loss = round(ori_loss.item(), 3)
        self.grad = delta.grad.detach().clone()
        
        # 根据mask_mode选择掩码生成方式
        if mask_mode in ['cam_lowr', 'cam_topr']:
            self.grayscale_cam, self.vis = run_grad_cam(
                self.model, self.images, self.labels, self.target_layers, 
                self.reshape_transform, self.use_cuda
            )
            mask, pixel_attacked = cam_mask(self.grayscale_cam, mode=mask_mode, **kwargs)
            
        elif mask_mode in ['lrp_lowr', 'lrp_topr'] and self.model_str == 'vgg16':
            lrp_model = LRPModel(self.model)
            relevance_scores = lrp_model.forward(self.images)
            mask, pixel_attacked = lrp_mask(relevance_scores, mode=mask_mode, **kwargs)
        
        else:
            mask, pixel_attacked = grad_mask(self.grad, mode=mask_mode, **kwargs)
        
        if show:
            titles = [f'{i+1}:{self.original_classes[i]}' for i in range(len(self.original_classes))]
            visualize_masks_overlay(self.images, mask, titles=titles, output_path=save_path, save_name='mask_overlay_visualization.png', nrows=self.nrows, ncols=self.ncols)
            visualize_gradients(self.grad, titles=titles, output_path=save_path, save_name='ori_grad_visualization.png', nrows=self.nrows, ncols=self.ncols)
            
            
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
        attack_ratio_per_channel = [round(x, 4) for x in attack_ratio_per_channel.tolist()]
        
        # 每个扰动下的预测损失: pred_loss = original_loss + grad * perturbation
        pred_loss_dict = {eta:original_loss + (perturbation * self.grad).sum().item() for eta, perturbation in zip(eta_list, perturbations)}
        
        success_rate_dict = {}
        loss_dict_attacked = {}  # 用于存储每个扰动下的攻击后实际的损失
        
        
        for i, eta in enumerate(eta_list):
            # 获取对应的对抗样本
            adv_image = self.adv_images[i].detach().clone().requires_grad_(True)
            
            # 计算对抗样本的输出和损失
            adv_output = self.model(adv_image)
            adv_loss = loss_fn(adv_output, self.labels)
            loss_attacked = adv_loss.item()  # 获取标量损失值
            loss_dict_attacked[eta] = round(loss_attacked, 6)  # 保存攻击后的损失,正数
            
            # 计算成功率
            pred = adv_output.argmax(dim=1)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[eta] = success_rate
    
            # 画图
            if show:
                adv_classes = get_classes_with_index(pred)
                
                titles = [f'{i+1}:{original}/{pred}' if original != pred else f'{i+1}:{original}' for i, (original, pred) in enumerate(zip(self.original_classes, adv_classes))]
                
                # main_title = f'{algo}, eta: {eta}, success_rate: {success_rate:.2f}, loss: {loss_dict_attacked[eta]}'
                main_title = None
                
                # 被攻击之后的图片
                show_images(self.adv_images[i], output_path=save_path, save_name=f'adv_images_eta{round(eta,2)}.png', titles=titles, main_title=main_title, nrows=self.nrows, ncols=self.ncols)
                # 扰动图像
                show_images(perturbations[i], output_path=save_path, save_name=f'perturbations_eta{round(eta,2)}.png', titles=titles, main_title=main_title, nrows=self.nrows, ncols=self.ncols)
                
                # 扰动分布以及一范数和二范数
                # norm1 = round(perturbations[i].abs().sum().cpu().item(), 4)
                # norm2 = round((perturbations[i] ** 2).sum().cpu().item(), 6)
                # main_title_per = f'eta: {eta}, L1: {norm1}, L2: {norm2}, success_rate: {success_rate:.2f}'
                # main_title_per = None
                # show_pixel_distribution(perturbations[i], output_path=save_path, save_name=f'perturbation_distribution_eta{round(eta,2)}.png', titles=titles, main_title=main_title_per,nrows=self.nrows, ncols=self.ncols)
                
                # 像素分布
                # show_pixel_distribution(adv_image, output_path=save_path,titles=titles, save_name=f'pixel_distribution_eta{round(eta,2)}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
                
                # Grad-CAM
                _, vis = run_grad_cam(self.model, self.adv_images[i], pred, self.target_layers, self.reshape_transform, self.use_cuda)
                show_images(vis, titles = titles, output_path=save_path, save_name=f'grad_cam_eta{round(eta,2)}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
                
                # 梯度可视化
                # 再次计算adv_image的梯度
                self.model.zero_grad()
                adv_loss.backward()
                grad = adv_image.grad.detach().clone()
                visualize_gradients(grad, titles=titles, output_path=save_path, save_name=f'adv_grad_visualization_eta{round(eta,2)}.png', nrows=self.nrows, ncols=self.ncols)
                
  
        return pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, loss_dict_attacked, pred_loss_dict, mask