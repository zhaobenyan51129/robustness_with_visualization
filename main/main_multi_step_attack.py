import time
import torch
import sys
import pandas as pd
from tqdm import tqdm
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from data_preprocessor.load_images import CustomDataset
from torch.utils.data import DataLoader
from algorithms.multi_step_attack import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
        
def parameter_total():
    algo_list = ['i_fgsm']
    eta_list = [0.01]
    alpha_list = [1e-4]
    steps = 300
    show = False
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': np.arange(0.05, 1, 0.05),
        'lowr': np.arange(0.05, 1, 0.05),
        'channel_topr': np.arange(0.05, 1, 0.05),
        'channel_lowr': np.arange(0.05, 1, 0.05),
        'randomr': np.arange(0.05, 1, 0.05),
        'seed_randomr': np.arange(0.05, 1, 0.05),
        'seed_randomr_lowr': np.arange(0.05, 1, 0.05),
        'cam_topr': np.arange(0.05, 1, 0.05),
        'cam_lowr': np.arange(0.05, 1, 0.05),
    }

    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    # model_list = ['vit_b_16']
    data_root = './data_stage3/multi_step_total100_1016'
    dataset_file = './data_stage2/images_100_0911.pth'
    save_result_file = 'result_multi_step_total100_1016.xlsx'
    return algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file, show

def parameter_vis():
    algo_list = ['i_fgsm']
    eta_list = [0.01]
    alpha_list = [1e-4]
    steps = 300
    show = True
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        # 'all': [None],
        'topr': [0.2],
        'lowr': [0.8],
        'channel_topr': [0.2],
        'channel_lowr': [0.8],
        # 'randomr':  [0.2],
        # 'seed_randomr': [0.2],
        # 'seed_randomr_lowr': [0.8],
        # 'cam_topr': [0.2],
        # 'cam_lowr': [0.8],
 }

    model_list = ['vit_b_16'] # ['vit_b_16', 'resnet50', 'vgg16']
    data_root = './data_stage3/vis_multi_step_1013'
    dataset_file = './data_stage2/images_100_0911.pth'
    save_result_file = 'vis_result_multi_step_1013.xlsx'
    return algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file, show

        
def main():
    algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file, show = parameter_total()
    
    # algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file, show = parameter_vis()
    
    print(f'data_root is {data_root}')
    
    dataset = CustomDataset(dataset_file)
    # dataset = torch.utils.data.Subset(dataset, range(16)) # 验证阶段，取16张
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # id_list = [0, 11, 3, 7, 8, 9, 10, 13]
    ncols = None
    nrows = None
    # dataset = torch.utils.data.Subset(dataset, id_list)
    # dataloader = DataLoader(dataset, batch_size=len(id_list), shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        results = pd.DataFrame(columns=['model', 'algo', 'alpha', 'mask_mode', 'step','parameter', 'eta', 'success_rate', 'l1_norm', 'l2_norm', 'loss', 'pred_loss', 'run_time', 'batch_idx', 'batch_pictures'])
        
        root = os.path.join(data_root, model_str)
        make_dir(root)
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1):
            batch_pictures = images.size(0)
            attacker = MultiStepAttack(model_str, images, labels, root, steps=steps,nrows=nrows, ncols=ncols)
            if show:
                titles = [f'{i+1}: {cls}' for i, cls in enumerate(get_classes_with_index(attacker.labels))]
                
                _, vis = run_grad_cam(attacker.model, attacker.images, attacker.labels, attacker.target_layers, attacker.reshape_transform, attacker.use_cuda)
                
                show_images(vis, titles = titles, output_path=root, save_name='ori_grad_cam.png', main_title= None, ncols=ncols, nrows=nrows)
                
                show_images(attacker.images, titles = titles, output_path=root, save_name='ori_images.png', main_title=None, ncols=ncols, nrows=nrows)
                
            for algo in algo_list:
                for eta in eta_list:
                    for alpha in alpha_list:
                        for mask_mode, parameters in mask_modes.items():
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
                                        **{mask_mode: parameter})
                                run_time = round(time.time() - start_time,3)

                                for step, success_rate in success_rate_dict.items():
                                    new_row = pd.DataFrame({
                                        'model': model_str, 
                                        'algo': algo, 
                                        'alpha': alpha, 
                                        'mask_mode': mask_mode, 
                                        'step': step, 
                                        'parameter': parameter, 
                                        'eta': eta, 
                                        'success_rate': success_rate, 
                                        'l1_norm': l1_norm_dict[step],
                                        'l2_norm': l2_norm_squre_dict[step],
                                        'loss': loss_dict[step],
                                        'pred_loss': pred_loss_dict[step],
                                        'run_time': run_time, 
                                        'batch_idx': batch_idx, 
                                        'batch_pictures': batch_pictures},
                                    index=[0])
                                    if results.empty:
                                        results = new_row
                                    else:
                                        results = pd.concat([results, new_row], ignore_index=True)
            torch.cuda.empty_cache()
        results.to_excel(os.path.join(root, f'{model_str}_{save_result_file}'), index=False)
    

if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
    







