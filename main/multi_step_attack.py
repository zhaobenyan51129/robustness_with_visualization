'''多步法'''
import time
import torch
import torch.nn as nn
import sys
import pandas as pd
from tqdm import tqdm
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
    
    def attack(self, algo, alpha, eta, mask_mode ='all', **kwargs):
        """ 对模型进行多步法攻击
        Args:
            algo: 攻击算法，str, 可选：'i_fgsm', 'gd' default: 'i_fgsm'
            alpha: 扰动的步长
            eta: 扰动阈值  
            mask_mode: 计算需要保留梯度的pixel，同单步法，str, 
                可选：'all', 'positive', 'negative', 'topr', 'lowr', 'randomr', 'cam_topr', 'cam_lowr', default: 'all'
        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
        """
        delta = torch.zeros_like(self.images, requires_grad=True)
        success_rate_dict = {}
        for t in tqdm(range(self.steps), desc="steps"):
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate
       
            loss = nn.CrossEntropyLoss()(output, self.labels)
            loss.backward()
            grad = delta.grad.detach().clone()
            if mask_mode == 'cam_topr':
                grayscale_cam, _ = run_grad_cam(self.model, self.images, self.labels, self.target_layers, self.reshape_transform, self.use_cuda)
                mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
            else:
                mask, _  = grad_mask(grad, mode=mask_mode, **kwargs)
            if algo == 'i_fgsm':
                delta.data = delta.data + alpha * mask * grad.sign()
            else:
                delta.data = delta.data + alpha * mask * grad
            delta.data = torch.clamp(delta.data, -eta, eta)
            delta.grad.zero_()  
            
        return delta, success_rate_dict
    
def parameter_sample():
    algo_list = ['i_fgsm']
    eta_list = [0.1]
    alpha_list = [0.01]
    steps= 10
    
    mask_modes = {
        # 'positive': [None],
        # 'negative': [None],
        'all': [None],
        # 'topr': [0.3], 
        # 'lowr': [0.3],
        # 'randomr': [0.3],
        # 'cam_topr': [0.5],  
        # 'cam_lowr': [0.5]
    }
    model_list = ['vit_b_16', 'resnet50', 'vgg16']
    data_root = './data_stage2/multi_step_sample_100_0912_test'
    return algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root
    
def main():
    results = pd.DataFrame(columns=['model', 'algo', 'alpha', 'mask_mode', 'step','parameter', 'eta', 'success_rate', 'run_time', 'batch_idx', 'batch_pictures'])

    algo_list, eta_list, alpha_list, steps, mask_modes, model_list, data_root = parameter_sample()
    print(f'data_root is {data_root}')
    
    dataset = CustomDataset('./data/images_100_0911.pth')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    for model_str in tqdm(model_list, desc="Models"):
        root = os.path.join(data_root, model_str)
        make_dir(root)
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1):
            batch_pictures = images.size(0)
            attacker = MultiStepAttack(model_str, images, labels, root, steps=steps)
            for algo in tqdm(algo_list, desc="Algorithms", leave=False):
                for mask_mode, mask_params in tqdm(mask_modes.items(), desc="Mask Modes", leave=False):
                    for eta in tqdm(eta_list, desc="Etas", leave=False):
                        for alpha in tqdm(alpha_list, desc="Alphas", leave=False):
                            for mask_param in mask_params:
                                start_time = time.time()
                                detla, success_rate_dict = attacker.attack(algo=algo, alpha=alpha, eta=eta, mask_mode=mask_mode, mask_param=mask_param)
                                run_time = time.time() - start_time
                                
                                for step, success_rate in success_rate_dict.items():
                                    new_row = pd.DataFrame({
                                        'model': model_str, 
                                        'algo': algo, 
                                        'alpha': alpha, 
                                        'mask_mode': mask_mode, 
                                        'step': step, 
                                        'parameter': mask_param, 
                                        'eta': eta, 
                                        'success_rate': success_rate, 
                                        'run_time': run_time, 
                                        'batch_idx': batch_idx, 
                                        'batch_pictures': batch_pictures},
                                    index=[0])
                                    if results.empty:
                                        results = new_row
                                    else:
                                        results = pd.concat([results, new_row], ignore_index=True)
            torch.cuda.empty_cache()
    results.to_excel(os.path.join(data_root, 'result_multi_step_sample100_0914.xlsx'), index=False)
    

if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'总共用时: {t1 - t0:.2f}秒')
    
    
    
    
    # model_str = 'vit_b_16'
    
    # root = './data/multi_step_attack_test'
    # make_dir(root)
    
    # dataset = CustomDataset('./data/images_100_0911.pth')
    # # 取16张图片测试
    # dataset = torch.utils.data.Subset(dataset, range(16))
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    # algp_list = ['i_fgsm']
    
    # for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Batches", leave=False), 1):
    #     attacker = MultiStepAttack(model_str, images, labels, root, steps=10)
    #     for algo in algp_list:
    #         delta, success_rate_dict = attacker.attack(algo=algo, alpha=0.01, eta=0.1, mask_mode='all')
    #         success_rate_df = pd.DataFrame(list(success_rate_dict.items()), columns=['Step', 'Success Rate'])
    #         success_rate_df.to_excel(os.path.join(root, 'success_rate_dict_{}.xlsx'.format(algo)), index=False)
    #         # print(delta.shape)
    #         # print('{} done!'.format(algo))
    







