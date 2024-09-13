import pandas as pd
import os
import torch
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from main.single_step_attack import *

def parameter_test():
    # 算法列表
    algo_list = ['fgsm', 'gaussian_noise'] # 删除了'fgm'
    eta_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    # eta_list = [0.01]
    mask_modes = {
        'positive': [None],
        'negative': [None],
        'all': [None],
        'topr': [0.1, 0.3, 0.5, 0.7, 0.9],
        'randomr': [0.1, 0.3, 0.5, 0.7, 0.9],
        'cam_topr': [0.1, 0.2, 0.3]
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    single_root = './data/one_step_attack_images_classified'
    make_dir(single_root)
    return algo_list, eta_list, mask_modes, model_list, single_root

# 选出来的类别
index_list = ['14', '900', '335', '75', '870', '50', '159', '793', '542', '675', '664']
show = False

# 对每一个类别进行攻击
algo_list, eta_list, mask_modes, model_list, single_root = parameter_test()
results = pd.DataFrame(columns=['index', 'model', 'algo', 'mask_mode', 'parameter', 'eta', 'num_attacked', 'numlocate', 'success_rate', 'picture_all', 'run_time'])
for index in index_list:
    for model_str in model_list:
        root = os.path.join(single_root, index, model_str)
        make_dir(root)
        
        image_path = f'./data/imges_classified/{index}.pth'
        images, labels = load_data(image_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)
        picture_all = len(labels)
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
                                    'index': [index],
                                    'model': [model_str],
                                    'algo': [algo],
                                    'mask_mode': [mask_mode],
                                    'parameter': [parameter],
                                    'eta': [eta],
                                    'num_attacked': [num_attacked],
                                    'numlocate': [numlocate],
                                    'success_rate': [success_rate],
                                    'picture_all': [picture_all],
                                    'run_time': [run_time]
                                })
                        if results.empty:
                            results = new_row
                        else:
                            results = pd.concat([results, new_row], ignore_index=True)
                    print(f'{index}, {model_str}, {algo}, {mask_mode}, {parameter} is finished!')
        torch.cuda.empty_cache()
results.to_excel(os.path.join(single_root, 'result_one_step_classified.xlsx'), index=False)

