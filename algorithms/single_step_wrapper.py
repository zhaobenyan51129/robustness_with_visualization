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
from tools.show_images import show_images
from models.load_model import load_model
from data_preprocessor.normalize import apply_normalization
from visualization.grad_cam import GradCAM, show_cam_on_image


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
    output = model(apply_normalization(X + delta))
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    grad = delta.grad.detach().clone()
    return grad

# -------------------- step2: 计算需要保留梯度的pixel --------------------
def grad_mask(grad, mode = None, **kwargs):
    '''对梯度进行掩码处理，生成于原始梯度相同形状的掩码，用于标记要修改的像素
    Args:mode
        'all': 返回全为1的掩码
        'positive': 梯度的正值为1，负值为0
        'negative': 梯度的负值为1，正值为0
        'topk':前k个梯度的绝对值为1，其余为0，需要传入参数k
        'topr':r为改变的pixel的比例，需要传入参数r
    '''
    if mode == 'all' or mode is None:
        mask = torch.ones_like(grad)
        num_attacked = mask.numel()
        return mask, num_attacked
    
    mode_dict = {
        'positive': grad_mask_positive,
        'negative': grad_mask_negative,
        'topk': grad_mask_topk,
        'topr': grad_mask_topr,
        'randomk': grad_mask_random,
        'randomr': grad_mask_randomr
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
    mask = - negative_grad.sign()
    return mask, num_negative

def grad_mask_topk(grad, topk = 10):
    '''topk为改变的pixel的个数，梯度绝对值前topk个为1，其余为0'''
    grad = grad.abs()
    top_array, _ = compute_top_indics(grad, topk)
    mask = torch.Tensor(top_array).to(device)
    return mask, topk

def grad_mask_topr(grad, topr = 0.1):
    '''topr为改变的pixel的比例，梯度绝对值前topr比例的为1，其余为0'''
    grad = grad.abs()
    num_pixels = grad[0].numel()
    num_change_pixels = int(num_pixels * topr)
    top_array, _ = compute_top_indics(grad, num_change_pixels)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels

def grad_mask_random(grad, randomk=10):
    '''随机选择randomk个像素，其余值置为0,并返回被修改的pixel数'''
    batch_size, _, height, width = grad.shape
    grad_view = grad.view(batch_size, -1)  
    mask = torch.zeros_like(grad_view)
    for i in range(batch_size):
        rand_indices = torch.randperm(grad_view.shape[1])[:randomk]
        mask[i, rand_indices] = 1  # set the selected pixels to 1
    mask = mask.view_as(grad)
    return mask, randomk

def grad_mask_randomr(grad, randomr=0.1):
    '''随机选择randomr比例的像素，其余值置为0,并返回被修改的pixel数'''
    batch_size, _, height, width = grad.shape
    grad_view = grad.view(batch_size, -1)  
    mask = torch.zeros_like(grad_view)
    for i in range(batch_size):
        num_change_pixels = int(grad_view.shape[1] * randomr)
        rand_indices = torch.randperm(grad_view.shape[1])[:num_change_pixels]
        mask[i, rand_indices] = 1  # set the selected pixels to 1
    mask = mask.view_as(grad)
    return mask, num_change_pixels

def cam_mask_topk(grayscale_cam, camtopk = 10):
    '''topk为改变的pixel的个数，cam绝对值前topk个为1，其余为0
    Args:
        grayscale_cam: cam图, 形状为[batch_size, height, width]
    '''
    cam = grayscale_cam.abs()
    # 将camtopk除以3并向上取整，因为cam是三通道的
    camtopk = int(camtopk / 3)
    top_array, _ = compute_top_indics(cam, camtopk)
    mask = torch.Tensor(top_array).to(device)
    return mask, camtopk

def cam_mask_topr(grayscale_cam, camtopr = 0.1):
    '''topr为改变的pixel的比例，cam绝对值前topr比例的为1，其余为0
    Args:
        grayscale_cam: cam图, 形状为[batch_size, height, width]
    '''
    cam = grayscale_cam.abs()
    num_pixels = cam[0].numel()
    num_change_pixels = int(num_pixels * camtopr) 
    top_array, _ = compute_top_indics(cam, num_change_pixels)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels

# -------------------- step3: 生成扰动 --------------------
def generate_perturbations(attack_method, eta_list, grad, **kwargs):
    '''生成扰动
    Args:
        attack_method: 攻击方法名称
        eta_list: 扰动的阈值
        grad: 梯度
    Returns:
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width], tensor
    '''
    attack_dict = {
        'fgsm': fgsm,
        'fgm': fgm,
        'gaussian_noise': gaussian_noise
    }
    perturbations = attack_dict[attack_method](eta_list, grad, **kwargs)
    return perturbations

def fgsm(eta_list, grad, **kwargs):
    '''Fast Gradient Sign Method
    Args:
        eta_list: 扰动的阈值
        grad: 梯度
    Returns:
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width], tensor
    '''
    perturbations = [eta * grad.sign() for eta in eta_list]
    return perturbations

def fgm(eta_list, grad, **kwargs):
    '''Fast Gradient Method'''
    batch_size = grad.shape[0]
    normed_grad =  torch.norm(grad.view(batch_size, -1), p=2, dim=1)
    perturbations = [eta * (grad / normed_grad.view(-1, 1, 1, 1)) for eta in eta_list]
    return perturbations

def gaussian_noise(eta_list, grad, **kwargs):
    '''高斯噪声
    fix:固定白噪声的pattern
        fix=True:先生成(0,1)正态分布standard_preturb,然后对每个eta,
        用eta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
        fix=False:每个eta都随机生成一个（0，eta）的噪声
    '''
    torch.manual_seed(0) # 固定随机种子
    fix = kwargs.get('fix', False)
    standard_perturb = torch.randn_like(grad)
    if fix:
        perturbations = [torch.clamp(eta * standard_perturb, -eta, eta) for eta in eta_list]
    else:
        perturbations = [torch.clamp(torch.randn_like(grad), -eta, eta) for eta in eta_list]
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
    # grad_masked = grad * mask
 
    perturbations = generate_perturbations('fgsm', eta_list, grad)
    perturbations = [perturbation * mask for perturbation in perturbations]
    adv_images = generate_adv_images(images, perturbations)
    for i, eta in enumerate(eta_list):
        pred = model(apply_normalization(adv_images[i])).argmax(dim=1)
        pred_classes = get_classes_with_index(pred)
        print(f'eta: {eta}, 攻击成功率: {(pred != labels).float().mean().item()}')
        titles = [f'{original} -> {pred}' if original != pred else original for original, pred in zip(original_classes, pred_classes)]
        show_images(adv_images[i], output_path = save_path, save_name = f'fgsm_{eta}.png', titles = titles)
    
    
        