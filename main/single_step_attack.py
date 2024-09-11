import time
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
        # 是否展示图片
        
        para = kwargs.get(mask_mode, None)
        if para is not None:
            save_path = os.path.join(self.root, algo, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, algo, mask_mode)
            
        if os.path.exists(save_path):
            self.show = False
        if mask_mode in ['cam_topk', 'cam_topr']:
            mask, num_attacked = cam_mask(self.grayscale_cam, mode=mask_mode, **kwargs)
        else:
            mask, num_attacked = grad_mask(self.grad, mode=mask_mode, **kwargs)
        
        sum_over_channels = mask.sum(dim=1)
        binary_mask = (sum_over_channels > 0).float()
        numlocate_per_sample = binary_mask.view(mask.size(0), -1).sum(dim=1)
        # numlocate = numlocate_per_sample[0].item()
        numlocate = int(numlocate_per_sample.mean().item())
        
        perturbations = generate_perturbations(algo, eta_list, self.grad)
        perturbations = [perturbation * mask for perturbation in perturbations]
        self.adv_images = generate_adv_images(self.images, perturbations)

        if self.show:
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
            
            if self.show:
                titles = [f'{original}/{pred}' if original != pred else original for original, pred in zip(self.original_classes, pred_classes)]
                main_title = f'{algo}, eta: {eta}, success_rate: {success_rate:.2f}'
            
                show_images(self.adv_images[i], output_path=save_path_image, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
                
                grayscale_cam, vis = run_grad_cam(self.model, self.adv_images[i], pred, self.target_layers, self.reshape_transform, self.use_cuda)
                show_images(vis, output_path=save_path_cam, save_name=f'{round(eta,2)}.png', titles=titles, main_title=main_title)
            
                perturbation = perturbations[i].permute(0, 2, 3, 1).detach().cpu().numpy()
                show_images(perturbation, output_path=save_path_perb, titles=titles, save_name=f'{round(eta,2)}.png', main_title=main_title)
                plot_distribution(perturbation, output_path=save_path_perb, save_name=f'distribution_{round(eta,2)}.png')
            
        return num_attacked, numlocate, success_rate_dict
    
def load_data(data_path):
    '''加载数据'''
    data = torch.load(data_path)
    images, labels = data['images'], data['labels']
    return images, labels

def parameter_sample():
    algo_list = ['fgsm']
    # eta_list = [0.01, 0.05, 0.1]
    eta_list = [0.1]
    
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topk': [10000],
        'topr': [0.3], 
        'randomk': [10000],
        'randomr': [0.3],
        'channel_randomk': [10000], # 不能超过224*224
        'channel_randomr': [0.3], # 不能超过1/3
        'cam_topk': [10000], # 不能超过224*224否则cam_mask会出错
        'cam_topr': [0.3]  # 不能超过1/3否则cam_mask会出错
    }
    model_list = ['vit_b_16']
    single_root = './data/one_step_attack_sample_900'
    return algo_list, eta_list, mask_modes, model_list, single_root
    
def parameter_test():
    # 算法列表
    algo_list = ['fgsm', 'gaussian_noise'] # 删除了'fgm'
    eta_list = np.arange(0.01, 0.31, 0.02)
    # mask_mode列表和对应的参数范围
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topk': range(1000, 20001, 1000),
        'topr': np.arange(0.05, 0.3, 0.05),
        'randomk': range(1000, 20001, 1000),
        'randomr': np.arange(0.05, 0.3, 0.05),
        'channel_randomk': range(1000, 20001, 1000),
        'channel_randomr': np.arange(0.05, 0.3, 0.05),
        'cam_topk': range(1000, 20001, 1000),
        'cam_topr': np.arange(0.05, 0.3, 0.05),
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    single_root = './data/one_step_attack'
    return algo_list, eta_list, mask_modes, model_list, single_root

def main():
    results = pd.DataFrame(columns=['model','algo', 'mask_mode', 'parameter', 'eta', 'num_attacked', 'numlocate', 'success_rate', 'run_time'])
    algo_list, eta_list, mask_modes, model_list, single_root = parameter_sample()
    for model_str in model_list:
        root = os.path.join(single_root, model_str)
        make_dir(root)
        
        # image_path = './data/images_new_100.pth'
        image_path = './data/images_900.pth'
        images, labels = load_data(image_path)
        attacker = OneStepAttack(model_str, images, labels, root, show)
        attacker.compute_grad_and_cam()
        # attacker.show_images_grad()
        
        for algo in algo_list:
            for mask_mode, parameters in mask_modes.items():
                for parameter in parameters:
                    # 记录运行时间
                    start_time = time.time()
                    if parameter is None:
                        num_attacked, numlocate, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode)  
                    else:
                        num_attacked, numlocate, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode, **{mask_mode: parameter}) 
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
                                    'num_attacked': [num_attacked],
                                    'numlocate': [numlocate],
                                    'success_rate': [success_rate],
                                    'run_time': [run_time]
                                })
                        if results.empty:
                            results = new_row
                        else:
                            results = pd.concat([results, new_row], ignore_index=True)
                    print(f'{model_str}, {algo}, {mask_mode}, {parameter} is finished!')
        torch.cuda.empty_cache()
    results.to_excel(os.path.join(single_root, 'result_one_step_sample_100.xlsx'), index=False)
    
def main_batch():
    results = pd.DataFrame(columns=['model', 'algo', 'mask_mode', 'parameter', 'eta', 'num_attacked', 'numlocate', 'success_rate', 'run_time', 'batch'])
    # algo_list, eta_list, mask_modes, model_list, single_root = parameter_sample()
    algo_list, eta_list, mask_modes, model_list, single_root = parameter_test()
    show = False
    batch_size = 100
    for model_str in model_list:
        image_path = './data/images_900.pth'
        images, labels = load_data(image_path)
        total_batches = len(images) // batch_size + (len(images) % batch_size > 0)
        for batch_idx in range(total_batches):
            root = os.path.join(single_root, model_str, str(batch_idx))
            if show:
                make_dir(root)
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(images))
            images_batch = images[start_idx:end_idx]
            labels_batch = labels[start_idx:end_idx]
            
            attacker = OneStepAttack(model_str, images_batch, labels_batch, root, show)
            attacker.compute_grad_and_cam()
            # attacker.show_images_grad()
     
            for algo in algo_list:
                for mask_mode, parameters in mask_modes.items():
                    for parameter in parameters:
                        start_time = time.time()
                        if parameter is None:
                            num_attacked, numlocate, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode)
                        else:
                            num_attacked, numlocate, success_rate_dict = attacker.attack(algo=algo, eta_list=eta_list, mask_mode=mask_mode, **{mask_mode: parameter})
                        end_time = time.time()
                        run_time = round(end_time - start_time, 3)
                        #  遍历返回的字典中的每一个eta和对应的成功率
                        for eta, success_rate in success_rate_dict.items():
                            # 将结果保存到DataFrame中
                            new_row = pd.DataFrame({
                                        'model': [model_str],
                                        'algo': [algo],
                                        'mask_mode': [mask_mode],
                                        'parameter': [parameter],
                                        'eta': [eta],
                                        'num_attacked': [num_attacked],
                                        'numlocate': [numlocate],
                                        'success_rate': [success_rate],
                                        'run_time': [run_time],
                                        'batch': [batch_idx]
                                    })
                            if results.empty:
                                results = new_row
                            else:
                                results = pd.concat([results, new_row], ignore_index=True)
                        print(f'{model_str}, {algo}, {mask_mode}, {parameter} , batch: {batch_idx} is finished!') 
            torch.cuda.empty_cache() 
    results.to_excel(os.path.join(single_root, 'result_one_step_900.xlsx'), index=False)
   

if __name__ == '__main__':
    # main()
    show = False 
    main_batch()
    
    