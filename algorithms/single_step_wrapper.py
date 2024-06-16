import numpy as np
import torch
import torch.nn as nn
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from tools.compute_topk import compute_top_indics
from tools.get_classes import get_classes_with_index
from tools.show_images import show_images, plot_distribution
from models.load_model import load_model


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'device: {device}')

# -------------------- step1: 计算梯度 --------------------
def compute_grad(model, X, y):
    '''计算梯度
    Args:
        model: 模型
        X: 图片
        y: 标签
    '''  
    model = model.to(device)
    model.eval()
    X = X.to(device)
    y = y.to(device)
    model.zero_grad()
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    output = model(X + delta)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    grad = delta.grad.detach().clone()
    return grad

# -------------------- step2: 计算需要保留梯度的pixel --------------------
def grad_mask(grad, mode = None, **kwargs):
    '''对梯度进行掩码处理，生成于原始梯度相同形状的掩码，用于标记要修改的像素
    Args:mode
        'None': 返回全为1的掩码
        'positive': 梯度的正值为1，负值为0
        'negative': 梯度的负值为1，正值为0
        'topk':前k个梯度的绝对值为1，其余为0，需要传入参数k
        'topr':r为改变的pixel的比例，需要传入参数r
    '''
    mode_dict = {
        'positive': grad_mask_positive,
        'negative': grad_mask_negative,
        'topk': grad_mask_topk,
        'topr': grad_mask_topr
    }
    mask, num_attacked = mode_dict[mode](grad, **kwargs)
    return mask, num_attacked

def grad_mask_positive(grad):
    '''只保留梯度的正值，负值置为0,并返回梯度为正的pixel数'''
    positive_grad = torch.clamp(grad, min=0)
    num_positive = positive_grad[positive_grad > 0].numel()
    mask = positive_grad.sign()
    return mask, num_positive

def grad_mask_negative(grad):
    '''只保留梯度的负值，正值置为0,并返回梯度为负的pixel数'''
    negative_grad = torch.clamp(grad, max=0)
    num_negative = negative_grad[negative_grad < 0].numel()
    mask = negative_grad.sign()
    return mask, num_negative

def grad_mask_topk(grad, k = 10):
    '''前k个梯度的绝对值为1，其余为0'''
    top_array, coordinates = compute_top_indics(grad, k)
    mask = torch.Tensor(top_array).to(device)
    return mask, k

def grad_mask_topr(grad, r = 0.1):
    '''r为改变的pixel的比例'''
    # mask = torch.zeros_like(grad)
    num_pixels = grad[0].numel()
    print(f'num_pixels: {num_pixels}')
    num_change_pixels = int(num_pixels * r)
    top_array, coordinates = compute_top_indics(grad, num_change_pixels)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels

# -------------------- step3: 生成扰动 --------------------
def generate_perturbations(attack_method, eta_list, masked_grad, **kwargs):
    '''生成扰动
    Args:
        attack_method: 攻击方法名称
        eta_list: 扰动的阈值
        masked_grad: 掩码后的梯度
    Returns:
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width], tensor
    '''
    attack_dict = {
        'fgsm': fgsm,
        'fgm': fgm,
        'gaussian_noise': gaussian_noise
    }
    perturbations = attack_dict[attack_method](eta_list, masked_grad, **kwargs)
    return perturbations

def fgsm(eta_list, masked_grad, **kwargs):
    '''Fast Gradient Sign Method
    Args:
        eta_list: 扰动的阈值
        masked_grad: 掩码后的梯度
    Returns:
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width], tensor
    '''
    perturbations = [eta * masked_grad.sign() for eta in eta_list]
    return perturbations

def fgm(eta_list, masked_grad, **kwargs):
    '''Fast Gradient Method'''
    batch_size = masked_grad.shape[0]
    normed_grad =  torch.norm(masked_grad.view(batch_size, -1), p=2, dim=1)
    perturbations = [eta * (masked_grad / normed_grad.view(-1, 1, 1, 1)) for eta in eta_list]
    return perturbations

def gaussian_noise(eta_list, masked_grad, **kwargs):
    '''高斯噪声
    fix:固定白噪声的pattern
        fix=True:先生成(0,1)正态分布standard_preturb,然后对每个eta,
        用eta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
        fix=False:每个eta都随机生成一个（0，eta）的噪声
    '''
    fix = kwargs.get('fix', False)
    standard_perturb = torch.randn_like(masked_grad)
    if fix:
        perturbations = [eta * standard_perturb for eta in eta_list]
    else:
        perturbations = [torch.clamp(torch.randn_like(masked_grad), -eta, eta) for eta in eta_list]
    return perturbations
    
# -------------------- step4: 生成对抗样本 --------------------
def generate_adv_images(images, perturbations):
    '''生成对抗样本
    Args:
        images: 原始图片, [batch_size, channel, height, width]
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width]
    '''
    if images.dim() == 4: # [batch_size, channel, height, width]
        images.unsqueeze(0) # [1, batch_size, channel, height, width]
    images = (images - images.min()) / (images.max() - images.min())
    adv_images = [(images + perturbation).to(device) for perturbation in perturbations]
    return adv_images
   
if __name__ == '__main__':
    model_str = 'vit_b_16'
    model = load_model(model_str)
    data = torch.load('./data/images_100.pth')
    images, labels = data['images'], data['labels']
    save_path = './data/one_step_attack'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_classes = get_classes_with_index(labels)
    # show_images(images, titles=original_classes, output_path = save_path, save_name = 'original.png')
    
    grad = compute_grad(model, images, labels)
    print(f'grad shape: {grad.shape}')
    # plot_distribution(grad, output_path = save_path, save_name = 'grad_distribution.png')
    mask, num_attacked = grad_mask(grad, mode = 'topr', r = 0.1)
    print(f'num_attacked: {num_attacked}')
    
    eta_list = np.arange(0, 0.2, 0.01)
    grad_masked = grad * mask
    print(f'grad_masked 大于0的个数: {grad_masked[grad_masked > 0].numel()}')
    print(f'grad_masked 小于0的个数: {grad_masked[grad_masked < 0].numel()}')
    perturbations = generate_perturbations('fgsm', eta_list, grad_masked)
    adv_images = generate_adv_images(images, perturbations)
    for i, eta in enumerate(eta_list):
        pred = model(adv_images[i]).argmax(dim=1)
        pred_classes = get_classes_with_index(pred)
        print(f'eta: {eta}, 攻击成功率: {(pred != labels).float().mean().item()}')
        titles = [f'{original} -> {pred}' if original != pred else original for original, pred in zip(original_classes, pred_classes)]
        show_images(adv_images[i], output_path = save_path, save_name = f'fgsm_{eta}.png', titles = titles)
    
    
        