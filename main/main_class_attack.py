import pandas as pd
import time
import os
import torch
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import time
from tqdm import tqdm
import multiprocessing

from tools.get_classes import get_classes_with_index
from algorithms.single_step_attack import make_dir, run_grad_cam
from tools.show_images import show_images
import numpy as np

def parameter_test_single():
    algo_list = ['fgsm'] 
    eta_list = [0.01]
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        # 'all': [None],
        'topr': np.arange(0.01, 1, 0.02), 
        'lowr': np.arange(0.01, 1, 0.02),
        'channel_topr': np.arange(0.01, 1, 0.02),
        'channel_lowr': np.arange(0.01, 1, 0.02),
        'seed_randomr': np.arange(0.01, 1, 0.02),
        'seed_randomr_lowr': np.arange(0.01, 1, 0.02),
        'cam_topr': np.arange(0.01, 1, 0.02),  
        'cam_lowr': np.arange(0.01, 1, 0.02), 
    }
    model = 'vit_b_16'
    data_root = './data_stage3/classified_single_attacktest_1109'
    make_dir(data_root)
    save_result_file = 'classified_single_attack_1109.xlsx'
    return algo_list, eta_list, mask_modes, model, data_root, save_result_file

def parameter_test_multi():
    algo = 'i_fgsm'
    eta = 0.01
    alpha = 2e-4
    steps = 100
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        # 'all': [None],
        'topr': [0.2],
        'lowr': [0.2],
        'channel_topr': [0.2],
        'channel_lowr': [0.2],
        # 'randomr':  [0.2],
        'seed_randomr': [0.2],
        'seed_randomr_lowr': [0.2],
        'cam_topr': [0.2],
        'cam_lowr': [0.2],
        # 'lrp_topr': [0.2],
        # 'lrp_lowr': [0.2],
    }

    model = 'vit_b_16'
    data_root = './data_stage3/classified_multi_attackall_1109'
    save_result_file = 'classified_multi_attack_1109.xlsx'
    return algo, eta, alpha, steps, mask_modes, model, data_root, save_result_file

def process_indices_single(indices, device_id, show):
    # 设置当前进程使用的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 在子进程中导入或初始化任何使用 CUDA 的模块或函数
    from data_preprocessor.load_images import CustomDataset
    from torch.utils.data import DataLoader
    from algorithms.single_step_attack import OneStepAttack

    algo_list, eta_list, mask_modes, model_str, data_root, _ = parameter_test_single()
    result_dir = os.path.join(data_root, 'results')
    make_dir(result_dir)
    
    for index in indices:
        results = pd.DataFrame(columns=['index', 'model', 'algo', 'mask_mode', 'parameter', 'eta', 'pixel_attacked', 'attack_ratio_per_channel', 'l1_norm', 'l2_norm', 'success_rate', 'original_loss', 'attack_loss', 'pred_loss','picture_all', 'picture_attacked', 'run_time'])
        dataset_file = f'./data_stage3/images_classified/{index}.pth'
        dataset = CustomDataset(dataset_file)
        picture_all = len(dataset)
        dataloader = DataLoader(dataset, batch_size=picture_all, shuffle=False)
        root = os.path.join(data_root, index)
        make_dir(root)
        
        for images, labels in dataloader:
            # 将数据移动到指定的设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 初始化攻击器，传入设备信息
            attacker = OneStepAttack(model_str, images, labels, root)
            if show:
                titles = [f'{i+1}: {cls}' for i, cls in enumerate(get_classes_with_index(attacker.labels))]
                _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
                show_images(vis, titles=titles, output_path=root, save_name='ori_grad_cam.png', main_title='Ori_Grad-CAM')
            
            for algo in tqdm(algo_list, desc="Algorithms", leave=False):
                for mask_mode, parameters in mask_modes.items():
                    if algo == 'gaussian_noise' and mask_mode not in ('all', 'positive', 'negative'):
                        continue
                    for parameter in parameters:
                        start_time = time.time()
                        if parameter is None:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, loss_dict_attacked, pred_loss_dict = attacker.attack(
                                algo=algo, eta_list=eta_list, show=show, mask_mode=mask_mode
                            )
                        else:
                            pixel_attacked, success_rate_dict, attack_ratio_per_channel, l1_norm, l2_norm_squre, original_loss, loss_dict_attacked, pred_loss_dict = attacker.attack(
                                algo=algo, eta_list=eta_list, show=show, mask_mode=mask_mode, **{mask_mode: parameter}
                            )
                        end_time = time.time()
                        run_time = round(end_time - start_time, 3)
                    
                        for eta, success_rate in success_rate_dict.items():
                            new_row = pd.DataFrame({
                                'index': [index],
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
                                'pred_loss': [pred_loss_dict[eta]],
                                'picture_all': [picture_all],
                                'picture_attacked': [int(picture_all * success_rate)],
                                'run_time': [run_time]
                            })
                            results = pd.concat([results, new_row], ignore_index=True)
        torch.cuda.empty_cache()
        # 保存每个 index 的结果
        results.to_excel(os.path.join(result_dir, f'results_{index}.xlsx'), index=False)
    
def process_indices_multi(indices, device_id, show):
    # 设置当前进程使用的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 在子进程中导入或初始化任何使用 CUDA 的模块或函数
    from data_preprocessor.load_images import CustomDataset
    from torch.utils.data import DataLoader
    from algorithms.multi_step_attack import MultiStepAttack

    algo, eta, alpha, steps, mask_modes, model_str, data_root, _ = parameter_test_multi()
    result_dir = os.path.join(data_root, 'results')
    make_dir(result_dir)

    for index in indices:
        results = pd.DataFrame(columns=['index', 'model', 'algo', 'alpha', 'mask_mode', 'parameter', 'step', 'eta',  'l1_norm', 'l2_norm','success_rate', 'loss', 'pred_loss', 'run_time'])
        dataset_file = f'./data_stage3/images_classified/{index}.pth'
        dataset = CustomDataset(dataset_file)
        picture_all = len(dataset)
        dataloader = DataLoader(dataset, batch_size=picture_all, shuffle=False)
        root = os.path.join(data_root, index)
        make_dir(root)
        
        for images, labels in dataloader:
            # 将数据移动到指定的设备
            images = images.to(device)
            labels = labels.to(device)
            
            attacker = MultiStepAttack(model_str, images, labels, root, steps=steps)
            if show:
                titles = [f'{i+1}: {cls}' for i, cls in enumerate(get_classes_with_index(attacker.labels))]
                _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
                show_images(vis, titles=titles, output_path=root, save_name='ori_grad_cam.png', main_title='Ori_Grad-CAM')
            
            for mask_mode, parameters in tqdm(mask_modes.items(), desc="Mask Modes", leave=False):
                for parameter in parameters:
                    start_time = time.time()
                    if parameter is None:
                        success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict, pred_loss_dict = attacker.attack(
                            algo=algo, 
                            alpha=alpha, 
                            eta=eta, 
                            mask_mode=mask_mode,
                            early_stopping=False,
                            show=show
                        )
                    else:
                        success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict, pred_loss_dict = attacker.attack(
                            algo=algo, 
                            alpha=alpha, 
                            eta=eta, 
                            mask_mode=mask_mode, 
                            early_stopping=False,
                            show=show,
                            **{mask_mode: parameter}
                        )
                    run_time = round(time.time() - start_time,3)
                    for step, success_rate in success_rate_dict.items():
                        new_row = pd.DataFrame({
                            'index': index,
                            'model': model_str,
                            'algo': algo,
                            'alpha': alpha,
                            'mask_mode': mask_mode,
                            'parameter': parameter,
                            'step': step,
                            'eta': eta,    
                            'l1_norm': l1_norm_dict[step],
                            'l2_norm': l2_norm_squre_dict[step],
                            'success_rate': success_rate,
                            'loss': loss_dict[step],
                            'pred_loss': pred_loss_dict[step],
                            'run_time': run_time,    
                        }, index=[0])
                        results = pd.concat([results, new_row], ignore_index=True)
        torch.cuda.empty_cache()
        # 保存每个 index 的结果
        results.to_excel(os.path.join(result_dir, f'results_{index}.xlsx'), index=False)

def main_single(index_list, show):
    num_gpus = torch.cuda.device_count()
    device_ids = [i for i in range(num_gpus)]
    
    _, _, _, _, data_root, save_result_file = parameter_test_single()
    result_dir = os.path.join(data_root, 'results')
    
    # 将 index_list 平均分配给每个 GPU
    index_chunks = [index_list[i::num_gpus] for i in range(num_gpus)]
    processes = []
    
    for device_id, indices in zip(device_ids, index_chunks):
        p = multiprocessing.Process(target=process_indices_single, args=(indices, device_id, show))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # 在主进程中合并所有结果文件
    combined_results = pd.DataFrame()
    for filename in os.listdir(result_dir):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(result_dir, filename)
            df = pd.read_excel(file_path)
            combined_results = pd.concat([combined_results, df], ignore_index=True)

    # 保存合并后的结果文件
    combined_results.to_excel(os.path.join(data_root, save_result_file), index=False)
    os.system(f'rm -r {result_dir}')
        
def main_multi(index_list, show):
    num_gpus = torch.cuda.device_count()
    device_ids = [i for i in range(num_gpus)]
    
    _, _, _, _, _, _, data_root, save_result_file = parameter_test_multi()
    result_dir = os.path.join(data_root, 'results')
    
    # 将 index_list 平均分配给每个 GPU
    index_chunks = [index_list[i::num_gpus] for i in range(num_gpus)]
    processes = []
    
    for device_id, indices in zip(device_ids, index_chunks):
        p = multiprocessing.Process(target=process_indices_multi, args=(indices, device_id, show))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # 在主进程中合并所有结果文件
    combined_results = pd.DataFrame()
    for filename in os.listdir(result_dir):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(result_dir, filename)
            df = pd.read_excel(file_path)
            combined_results = pd.concat([combined_results, df], ignore_index=True)

    # 保存合并后的结果文件
    combined_results.to_excel(os.path.join(data_root, save_result_file), index=False)
    os.system(f'rm -r {result_dir}')
    

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # mode = 'single' 
    mode = 'multi' 
    show = False
    # 选出来的类别
    index_list = ['110', '174', '230', '241', '249', '254', '369', '408', '423', '460', '492', '534', '552', '723', '725', '733', '741', '751', '848', '948']
    # index_list = ['552']
    
    t0 = time.time()
    if mode == 'single':
        main_single(index_list, show)
    else:
        main_multi(index_list, show)
    
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
