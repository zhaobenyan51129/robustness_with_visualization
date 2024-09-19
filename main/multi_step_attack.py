'''多步法'''
import time
import torch
import torch.nn as nn
import sys
import pandas as pd
from tqdm import tqdm
from collections import deque
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from data_preprocessor.load_images import CustomDataset
from torch.utils.data import DataLoader
from algorithms.single_step_wrapper import *
from main.single_step_attack import OneStepAttack, run_grad_cam, make_dir

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MultiStepAttack(OneStepAttack):
    def __init__(self, model_str, images, labels, root, steps=10):
        super().__init__(model_str, images, labels, root, show = False)
        self.steps = steps
    
    def compute_loss_function(self, output, y, loss_mode):
        '''计算损失函数
        Args:
            output: 模型输出，tensor, [batch_size, num_classes]
            y: 真实标签，tensor, [batch_size,]
            loss_mode: 损失函数的模式，str, 可选：'CrossEntropy'，'logit', 'softmax', default: 'CrossEntropy'
        Returns:
            loss: 损失函数值，tensor
        '''
        if loss_mode == 'CrossEntropy':
            loss = nn.CrossEntropyLoss()(output, y)
        elif loss_mode == 'logit':
            loss = get_loss(output, y)
        elif loss_mode == 'softmax':
            output = nn.Softmax(dim=1)(output)
            loss = get_loss(output, y)
        return loss
    
    def attack(self, algo, alpha, eta, mask_mode='all', early_stopping=False, patience=50, tol=0.01, target_success_rate=1.0, **kwargs):
        """
        对模型进行多步法攻击，增加早停策略。

        Args:
            algo: 攻击算法，str, 可选：'i_fgsm', 'gd' default: 'i_fgsm'
            alpha: 扰动的步长
            eta: 扰动阈值  
            mask_mode: 计算需要保留梯度的pixel，同单步法，str, 
                可选：'all', 'positive', 'negative', 'topr', 'lowr', 'randomr', 'cam_topr', 'cam_lowr', default: 'all'
            early_stopping: 是否启用早停策略，bool, default: False
            patience: 在没有显著提升的情况下，允许的最大连续不改进的步数，int, default: 50
            tol: 损失值下降的最小幅度，float, default: 0.01
            target_success_rate: 达到该成功率时，提前终止攻击，float, default: 1.0
            **kwargs: 其他可选参数

        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
            success_rate_dict: dict记录每步的攻击成功率
            loss_dict: dict记录每步的损失值
            l1_norm_dict: dict记录每步的L1范数
            l2_norm_square_dict: dict记录每步的L2范数平方
        """
        # 初始化扰动
        delta = torch.zeros_like(self.images, requires_grad=True)
        
        # 初始化记录字典
        success_rate_dict = {}
        loss_dict = {}
        l1_norm_dict = {}
        l2_norm_square_dict = {}
        
        if early_stopping:
            # 使用deque来存储最近patience步的成功率和损失值
            success_rates_window = deque(maxlen=patience)
            losses_window = deque(maxlen=patience)
            
            # 记录迄今为止的最佳成功率和最佳损失值
            best_success_rate = 0.0
            best_loss = float('inf')
            
            # 记录连续未改进的步数
            no_improve_steps_success = 0
            no_improve_steps_loss = 0

        for t in tqdm(range(self.steps), desc="steps"):
            # 前向传播
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            
            # 计算攻击成功率
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate

            # 计算损失值（交叉熵损失）
            loss = nn.CrossEntropyLoss()(output, self.labels)
            loss_dict[t] = - loss.item()
            loss.backward()
            grad = delta.grad.detach().clone()
            
            # 根据mask_mode生成掩码
            if mask_mode in ('cam_topr', 'cam_lowr'):
                grayscale_cam, _ = run_grad_cam(
                    self.model, self.images, self.labels, 
                    self.target_layers, self.reshape_transform, self.use_cuda
                )
                mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
            else:
                mask, _ = grad_mask(grad, mode=mask_mode, **kwargs)
            
            # 计算被攻击pixel的梯度范数
            masked_grad = grad * mask
            l1_norm = round(masked_grad.abs().sum().cpu().item(), 4)
            l2_norm_square = round((masked_grad ** 2).sum().cpu().item(), 6)
            l1_norm_dict[t] = l1_norm
            l2_norm_square_dict[t] = l2_norm_square

            # 更新扰动
            if algo == 'i_fgsm':
                delta.data = delta.data + alpha * mask * grad.sign()
            else:
                delta.data = delta.data + alpha * mask * grad
            delta.data = torch.clamp(delta.data, -eta, eta)
            delta.grad.zero_()
            
            # 早停策略
            if early_stopping:
                # 将当前的成功率和损失值添加到窗口中
                success_rates_window.append(success_rate)
                losses_window.append(-loss.item())
                
                # 检查是否达到目标成功率
                if success_rate >= target_success_rate:
                    print(f"步数 {t}: 达到目标成功率 {target_success_rate*100}%，提前终止攻击。")
                    break

                # 检查最近patience步内是否有成功率提升
                if any(sr > best_success_rate for sr in success_rates_window):
                    best_success_rate = max(success_rates_window)
                    no_improve_steps_success = 0
                else:
                    no_improve_steps_success += 1
                    if no_improve_steps_success >= patience:
                        print(f"步数 {t}: 在连续 {patience} 步中，攻击成功率没有提升提前终止攻击, 参数：model_str:{self.model_str}, algo:{algo}, alpha:{alpha}, eta:{eta}, mask_mode:{mask_mode}")
                        break
                
                # 检查最近patience步内是否有损失值显著下降
                min_recent_loss = min(losses_window) if losses_window else float('inf')
                if (min_recent_loss - best_loss)/abs(min_recent_loss) < tol:
                    best_loss = min_recent_loss
                    no_improve_steps_loss = 0
                else:
                    no_improve_steps_loss += 1
                    if no_improve_steps_loss >= patience:
                        print(f"步数 {t}: 在连续 {patience} 步中，损失值没有显著下降。提前终止攻击。")
                        break
        return delta, success_rate_dict, loss_dict, l1_norm_dict, l2_norm_square_dict
        
def parameter_sample():
    algo_list = ['i_fgsm']
    eta_list = [0.01]
    alpha_list = [1e-4]
    steps = 500
    
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
    # mask_modes = {
    #     'positive': [None],
    #     'negative': [None],
    #     # 'all': [None],
    #     'topr': [0.3],
    #     'lowr': [0.3],
    #     'randomr':  [0.3],
    #     # 'channel_randomr': np.arange(0.05, 0.3, 0.05),
    #     'cam_topr': [0.3],
    #     'cam_lowr': [0.3],
    # }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    # model_list = ['vgg16']
    data_root = './data_stage2/multi_step_total100_0918'
    dataset_file = './data/images_100_0911.pth'
    save_result_file = 'result_multi_step_total100_0918.xlsx'
    return algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file
        
def main():
    algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root, dataset_file, save_result_file = parameter_sample()
    print(f'data_root is {data_root}')
    
    dataset = CustomDataset(dataset_file)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        results = pd.DataFrame(columns=['model', 'algo', 'alpha', 'mask_mode', 'step','parameter', 'eta', 'success_rate', 'l1_norm', 'l2_norm', 'run_time', 'batch_idx', 'batch_pictures'])
        
        root = os.path.join(data_root, model_str)
        make_dir(root)
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1):
            batch_pictures = images.size(0)
            attacker = MultiStepAttack(model_str, images, labels, root, steps=steps)
            for algo in algo_list:
                for eta in eta_list:
                    for alpha in alpha_list:
                        for mask_mode, parameters in tqdm(mask_modes.items(), desc="Mask Modes", leave=False):
                            for parameter in parameters:
                                start_time = time.time()
                                if parameter is None:
                                   detla, success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict = attacker.attack(
                                       algo=algo, 
                                       alpha=alpha, 
                                       eta=eta, 
                                       mask_mode=mask_mode,
                                       early_stopping=True)
                                else:
                                    detla, success_rate_dict, loss_dict, l1_norm_dict, l2_norm_squre_dict = attacker.attack(
                                        algo=algo, 
                                        alpha=alpha, 
                                        eta=eta, 
                                        mask_mode=mask_mode, 
                                        early_stopping=True,
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
                                        'run_time': run_time, 
                                        'batch_idx': batch_idx, 
                                        'batch_pictures': batch_pictures},
                                    index=[0])
                                    if results.empty:
                                        results = new_row
                                    else:
                                        results = pd.concat([results, new_row], ignore_index=True)
            torch.cuda.empty_cache()
        results.to_excel(os.path.join(data_root, f'{model_str}_{save_result_file}'), index=False)
    

if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
    







