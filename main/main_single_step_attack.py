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
from algorithms.single_step_attack import *
from data_preprocessor.load_images import CustomDataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def parameter_vis():
    algo_list = ['fgsm', 'gaussian_noise', 'gaussian_noise_sign','gaussian_noise_std', 'gaussian_noise_sign_std']
    eta_list = [0.01]
    show = True
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        'all': [None],
        # 'topr': [0.15], 
        # 'lowr': [0.85],
        # 'randomr': [0.15],
        # 'channel_randomr': [0.3], # 不能超过1/3
        # 'cam_topr': [0.15],  
        # 'cam_lowr': [0.85], 
    }
    # model_list = ['vit_b_16', 'resnet50', 'vgg16']
    model_list = ['vit_b_16']
    data_root = './data_stage2/vis_single_step_attack_16'
    dataset_file = './data_stage2/images_100_0911.pth'
    save_result_file = 'vis_one_step_sample100_0922.xlsx'
    return algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file, show
    
def parameter_total():
    algo_list = ['fgsm', 'gaussian_noise']
    eta_list = np.arange(0.005, 0.105, 0.005)
    show = False
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
    return algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file, show

def main():
    results = pd.DataFrame(columns=['model', 'algo', 'mask_mode', 'parameter', 'eta', 'pixel_attacked', 'attack_ratio_per_channel', 'l1_norm', 'l2_norm','success_rate', 'original_loss', 'loss_dict_attacked', 'run_time', 'batch_idx', 'batch_pictures'])
    algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file, show = parameter_vis()
    # algo_list, eta_list, mask_modes, model_list, data_root, dataset_file, save_result_file, show = parameter_total()
    print(f'data_root is {data_root}')

    dataset = CustomDataset(dataset_file)
    dataset = torch.utils.data.Subset(dataset, range(16)) # 验证阶段，取16张
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        root = os.path.join(data_root, model_str)
        make_dir(root)
        # 遍历 DataLoader
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1): # 1表示从1开始计数
            batch_pictures = images.size(0)
            attacker = OneStepAttack(model_str, images, labels, root)
            # if show:
            #     _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
            #     show_images(vis, titles = get_classes_with_index(attacker.labels), output_path=root, save_name='ori_grad_cam.png', main_title='Ori_Grad-CAM')
            
            for algo in algo_list:
                for mask_mode, parameters in mask_modes.items():
                    if algo == 'gaussian_noise' and mask_mode not in ('all', 'positive', 'negative'):
                        continue
                    for parameter in tqdm(parameters, desc="Parameters", leave=False):
                        # 记录运行时间
                        start_time = time.time()
                        if parameter is None:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, loss_dict_attacked = attacker.attack(algo=algo, eta_list=eta_list, show=show, mask_mode=mask_mode)  
                        else:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, loss_dict_attacked = attacker.attack(algo=algo, eta_list=eta_list, show=show, mask_mode=mask_mode, **{mask_mode: parameter}) 
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
                                        'attack_loss': [loss_dict_attacked[eta]],
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
    
    