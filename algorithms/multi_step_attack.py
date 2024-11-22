import torch
import torch.nn as nn
import sys
from collections import deque
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from algorithms.single_step_wrapper import *
from tools.get_classes import get_classes_with_index
from algorithms.single_step_attack import OneStepAttack, run_grad_cam, make_dir
from tools.show_images import show_images, visualize_masks_overlay, visualize_gradients
from algorithms.LRP.lrp import LRPModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MultiStepAttack(OneStepAttack):
    def __init__(self, model_str, images, labels, root, steps=10, **kwargs):
        super().__init__(model_str, images, labels, root, **kwargs)
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
    
    def attack(self, algo, alpha, eta, mask_mode='all', early_stopping=False, patience=200, tol=0.01, target_success_rate=1.0, show = False, **kwargs):
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
            show: 是否画图，bool, default: False
            **kwargs: 其他可选参数

        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
            success_rate_dict: dict记录每步的攻击成功率
            loss_dict: dict记录每步的损失值
            l1_norm_dict: dict记录每步的L1范数
            l2_norm_square_dict: dict记录每步的L2范数平方
        """
        # 获取掩码参数并构建保存路径
        para = kwargs.get(mask_mode, None)
    
        if para is not None:
            save_path = os.path.join(self.root, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, mask_mode)
            
        # 初始化扰动
        delta = torch.zeros_like(self.images, requires_grad=True)
        
        # 初始化记录字典
        success_rate_dict = {}
        loss_dict = {}
        pred_loss_dict = {}
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

        for t in range(self.steps):
            # 前向传播
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            
            # 计算攻击成功率
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate

            # 计算损失值（交叉熵损失）
            loss = nn.CrossEntropyLoss()(output, self.labels)
            loss_dict[t] = round(loss.item(), 6)
            if t == 0:
                original_loss = loss_dict[t]
            loss.backward()
            grad = delta.grad.detach().clone()
            
            pred_loss_dict[t] = original_loss + (delta * grad).sum().item()
            
            # 根据mask_mode生成掩码
            if mask_mode in ('cam_topr', 'cam_lowr'):
                grayscale_cam, _ = run_grad_cam(
                    self.model, self.images, self.labels, 
                    self.target_layers, self.reshape_transform, self.use_cuda
                )
                mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
            elif mask_mode in ['lrp_lowr', 'lrp_topr']:
                if self.model_str == 'vgg16':
                    lrp_model = LRPModel(self.model)
                    relevance_scores = lrp_model.forward(self.images)
                    mask, pixel_attacked = lrp_mask(relevance_scores, mode=mask_mode, **kwargs)
                else:
                    break
            
            else:
                mask, _ = grad_mask(grad, mode=mask_mode, **kwargs)
            
            # 每隔100步画一次mask
            # if show and t % 100 == 0:
            #     adv_classes = get_classes_with_index(pred)
            #     titles = [f'{i+1}:{original}/{pred}' if original != pred else f'{i+1}:{original}' for i, (original, pred) in enumerate(zip(self.original_classes, adv_classes))]
            #     main_title = f'success_rate: {success_rate:.2f}, loss: {loss_dict[t]:.4f}'
                # visualize_masks_overlay(self.images, mask, titles=titles, output_path=save_path, main_title = main_title, save_name=f'mask_overlay_visualization_step{t}.png', nrows=self.nrows, ncols=self.ncols)
            if show and t == 0:
                # 原始梯度
                visualize_gradients(grad, output_path=save_path, save_name=f'ori_gradient.png', main_title=None, titles = get_classes_with_index(self.labels), nrows=self.nrows, ncols=self.ncols)
            
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
        if show: # 画出最后一步的可视化结果
            adv_classes = get_classes_with_index(pred)
            titles = [f'{i+1}:{original}/{pred}' if original != pred else f'{i+1}:{original}' for i, (original, pred) in enumerate(zip(self.original_classes, adv_classes))]
            
            # main_title = f'{algo}, eta: {eta}, success_rate: {success_rate:.2f}, loss: {loss_dict[t]:.4f}'
            main_title = None
            
            # 扰动delta
            show_images(delta, titles=titles, output_path=save_path, save_name=f'delta_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # 扰动后的图片
            show_images(self.images + delta, titles=titles, output_path=save_path, save_name=f'adversarial_images_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # Grad-CAM
            _, vis = run_grad_cam(self.model, self.images + delta, pred, self.target_layers, self.reshape_transform, self.use_cuda)
            show_images(vis, titles = titles, output_path=save_path, save_name=f'grad_cam_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # 梯度
            visualize_gradients(grad, output_path=save_path, save_name=f'gradient_step{t}.png', main_title=main_title, titles = titles, nrows=self.nrows, ncols=self.ncols)
            # mask
            visualize_masks_overlay(self.images, mask, titles=titles, output_path=save_path, main_title = main_title, save_name=f'mask_overlay_visualization_step{t}.png', nrows=self.nrows, ncols=self.ncols)
            
            
        return success_rate_dict, loss_dict, l1_norm_dict, l2_norm_square_dict, pred_loss_dict

    def attack_fixed(self, algo, alpha, eta, mask_mode='all', early_stopping=False, patience=200, tol=0.01, target_success_rate=1.0, show=False, **kwargs):
        """
        对模型进行多步法攻击，第一步确定攻击位置后固定。

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
            show: 是否画图，bool, default: False
            **kwargs: 其他可选参数

        Returns:
            delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor
            success_rate_dict: dict记录每步的攻击成功率
            loss_dict: dict记录每步的损失值
            l1_norm_dict: dict记录每步的L1范数
            l2_norm_square_dict: dict记录每步的L2范数平方
        """
        # 获取掩码参数并构建保存路径
        para = kwargs.get(mask_mode, None)
        
        if para is not None:
            save_path = os.path.join(self.root, mask_mode, str(para))
        else:
            save_path = os.path.join(self.root, mask_mode)
                
        # 初始化扰动
        delta = torch.zeros_like(self.images, requires_grad=True)
        
        # 初始化记录字典
        success_rate_dict = {}
        loss_dict = {}
        pred_loss_dict = {}
        l1_norm_dict = {}
        l2_norm_square_dict = {}
        
        # 计算初始掩码
        output = self.model(self.images + delta)
        pred = output.argmax(dim=1)
        loss = nn.CrossEntropyLoss()(output, self.labels)
        loss.backward()
        grad = delta.grad.detach().clone()
        
        if mask_mode in ('cam_topr', 'cam_lowr'):
            grayscale_cam, _ = run_grad_cam(
                self.model, self.images, self.labels,
                self.target_layers, self.reshape_transform, self.use_cuda
            )
            mask, _ = cam_mask(grayscale_cam, mode=mask_mode, **kwargs)
        elif mask_mode in ['lrp_lowr', 'lrp_topr']:
            if self.model_str == 'vgg16':
                lrp_model = LRPModel(self.model)
                relevance_scores = lrp_model.forward(self.images)
                mask, pixel_attacked = lrp_mask(relevance_scores, mode=mask_mode, **kwargs)
            else:
                raise ValueError("LRP mask mode is only supported for 'vgg16' model.")
        else:
            mask, _ = grad_mask(grad, mode=mask_mode, **kwargs)
        
        # 固定掩码
        fixed_mask = mask.clone().detach()
        
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

        for t in range(self.steps):
            # 前向传播
            output = self.model(self.images + delta)
            pred = output.argmax(dim=1)
            
            # 计算攻击成功率
            success_rate = (pred != self.labels).float().mean().item()
            success_rate_dict[t] = success_rate

            # 计算损失值（交叉熵损失）
            loss = nn.CrossEntropyLoss()(output, self.labels)
            loss_dict[t] = round(loss.item(), 6)
            if t == 0:
                original_loss = loss_dict[t]
            loss.backward()
            grad = delta.grad.detach().clone()
            
            pred_loss_dict[t] = original_loss + (delta * grad).sum().item()
            
            # 使用固定掩码
            masked_grad = grad * fixed_mask
            l1_norm = round(masked_grad.abs().sum().cpu().item(), 4)
            l2_norm_square = round((masked_grad ** 2).sum().cpu().item(), 6)
            l1_norm_dict[t] = l1_norm
            l2_norm_square_dict[t] = l2_norm_square

            # 更新扰动
            if algo == 'i_fgsm':
                delta.data = delta.data + alpha * fixed_mask * grad.sign()
            else:
                delta.data = delta.data + alpha * fixed_mask * grad
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
        if show:  # 画出最后一步的可视化结果
            adv_classes = get_classes_with_index(pred)
            titles = [f'{i+1}:{original}/{pred}' if original != pred else f'{i+1}:{original}' for i, (original, pred) in enumerate(zip(self.original_classes, adv_classes))]
            
            main_title = None
            
            # 扰动delta
            show_images(delta, titles=titles, output_path=save_path, save_name=f'delta_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # 扰动后的图片
            show_images(self.images + delta, titles=titles, output_path=save_path, save_name=f'adversarial_images_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # Grad-CAM
            _, vis = run_grad_cam(self.model, self.images + delta, pred, self.target_layers, self.reshape_transform, self.use_cuda)
            show_images(vis, titles=titles, output_path=save_path, save_name=f'grad_cam_step{t}.png', main_title=main_title, nrows=self.nrows, ncols=self.ncols)
            # 梯度
            visualize_gradients(grad, output_path=save_path, save_name=f'gradient_step{t}.png', main_title=main_title, titles=titles, nrows=self.nrows, ncols=self.ncols)
            # mask
            visualize_masks_overlay(self.images, fixed_mask, titles=titles, output_path=save_path, main_title=main_title, save_name=f'mask_overlay_visualization_step{t}.png', nrows=self.nrows, ncols=self.ncols)
            
        return success_rate_dict, loss_dict, l1_norm_dict, l2_norm_square_dict, pred_loss_dict