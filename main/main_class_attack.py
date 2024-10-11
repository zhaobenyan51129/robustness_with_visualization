import pandas as pd
import time
import os
import torch
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import time
from tqdm import tqdm
from data_preprocessor.load_images import CustomDataset
from torch.utils.data import DataLoader
from algorithms.single_step_attack import *
from algorithms.multi_step_attack import *

def parameter_test_single():
    algo_list = ['fgsm'] 
    eta_list = [0.01]
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': [0.15], 
        'lowr': [0.85],
        'randomr': [0.15],
        'cam_topr': [0.15],  
        'cam_lowr': [0.85], 
    }
    model = 'vit_b_16'
    data_root = './data_stage2/classified_single_attack'
    make_dir(data_root)
    save_result_file = 'classified_single_attack_1010.xlsx'
    return algo_list, eta_list, mask_modes, model, data_root, save_result_file

def parameter_test_multi():
    algo = 'i_fgsm'
    eta = 0.01
    alpha = 1e-4
    steps = 300
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': [0.2],
        'lowr': [0.8],
        'randomr':  [0.2],
        'cam_topr': [0.2],
        'cam_lowr': [0.8],
    }

    model = 'vit_b_16'
    data_root = './data_stage2/classified_multi_attack'
    save_result_file = 'classified_multi_attack_1011.xlsx'
    return algo, eta, alpha, steps, mask_modes, model, data_root, save_result_file

def main_single(index_list, show):
    algo_list, eta_list, mask_modes, model_str, data_root, save_result_file = parameter_test_single()

    results = pd.DataFrame(columns=['index', 'model', 'algo', 'mask_mode', 'parameter', 'eta', 'pixel_attacked', 'attack_ratio_per_channel', 'l1_norm', 'l2_norm','success_rate', 'original_loss', 'loss_dict_attacked', 'picture_all', 'picture_attacked', 'run_time'])

    for index in tqdm(index_list, desc="Index", leave=False):
        dataset_file = f'./data_stage2/images_classified/{index}.pth'
        dataset = CustomDataset(dataset_file)
        picture_all = len(dataset)
        dataloader = DataLoader(dataset, batch_size=picture_all, shuffle=False)
        root = os.path.join(data_root,index)
        
        for images, labels in dataloader: 
            attacker = OneStepAttack(model_str, images, labels, root)
            if show:
                titles = [f'{i+1}: {cls}' for i, cls in get_classes_with_index(attacker.labels)]
        
                _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
                show_images(vis, titles = titles, output_path=root, save_name='ori_grad_cam.png', main_title='Ori_Grad-CAM')
            
            for algo in tqdm(algo_list, desc="Algorithms", leave=False):
                for mask_mode, parameters in mask_modes.items():
                    if algo == 'gaussian_noise' and mask_mode not in ('all', 'positive', 'negative'):
                        continue
                    for parameter in parameters:
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
                                        'loss_dict_attacked': [loss_dict_attacked[eta]],
                                        'picture_all': [picture_all],
                                        'picture_attacked':[int(picture_all * success_rate)],
                                        'run_time': [run_time]
                                    })
                            if results.empty:
                                results = new_row
                            else:
                                results = pd.concat([results, new_row], ignore_index=True)
            torch.cuda.empty_cache()
    results.to_excel(os.path.join(data_root, save_result_file), index=False)
    
def main_multi(index_list, show):
    algo, eta, alpha, steps, mask_modes, model_str, data_root, save_result_file = parameter_test_multi()

    results = pd.DataFrame(columns=['index', 'model', 'algo', 'alpha', 'mask_mode', 'parameter', 'step', 'eta',  'l1_norm', 'l2_norm','success_rate', 'loss', 'run_time'])

    for index in tqdm(index_list, desc="Index", leave=False):
        dataset_file = f'./data_stage2/images_classified/{index}.pth'
        dataset = CustomDataset(dataset_file)
        picture_all = len(dataset)
        dataloader = DataLoader(dataset, batch_size=picture_all, shuffle=False)
        root = os.path.join(data_root, index)
        
        for images, labels in dataloader: 
            attacker = MultiStepAttack(model_str, images, labels, root, steps=steps)
            if show:
                titles = [f'{i+1}: {cls}' for i, cls in get_classes_with_index(attacker.labels)]
        
                _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
                show_images(vis, titles = titles, output_path=root, save_name='ori_grad_cam.png', main_title='Ori_Grad-CAM')
                
            for mask_mode, parameters in tqdm(mask_modes.items(), desc="Mask Modes", leave=False):
                for parameter in parameters:
                    start_time = time.time()
                    if parameter is None:
                        success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict = attacker.attack(
                            algo=algo, 
                            alpha=alpha, 
                            eta=eta, 
                            mask_mode=mask_mode,
                            early_stopping=True,
                            show=show
                            )
                    else:
                        success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict = attacker.attack(
                            algo=algo, 
                            alpha=alpha, 
                            eta=eta, 
                            mask_mode=mask_mode, 
                            early_stopping=False,
                            show=show,
                            **{mask_mode: parameter})
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
                            'run_time': run_time,    
                           },
                        index=[0])
                        if results.empty:
                            results = new_row
                        else:
                            results = pd.concat([results, new_row], ignore_index=True)
        torch.cuda.empty_cache()
    results.to_excel(os.path.join(root, f'{model_str}_{save_result_file}'), index=False)


if __name__ == '__main__':
    mode = 'multi' # 'single'
    show = False
    # 选出来的类别
    # index_list = ['1', '512', '569', '642', '959', '680', '314', '468', '382', '460', '782']
    index_list = ['1']
    
    t0 = time.time()
    if mode == 'single':
        main_single(index_list, show)
    else:
        main_multi(index_list, show)
    
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
