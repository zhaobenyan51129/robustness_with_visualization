'''单步法'''
import numpy as np
import torch
import torch.nn as nn
# from torch.autograd.gradcheck import zero_gradients
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from tools.compute_topk import compute_top_indics
from tools.get_classes import get_classes_with_index
from models.load_model import load_model
# from data_preprocessor.normalize import apply_normalization


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# -------------------- step1: 计算梯度，已弃用 --------------------
def compute_grad(model, X, y):
    '''计算loss对输入的梯度, 对x和对delta求梯度结果是一样的
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
    return grad, loss.item()

def get_loss(output, target_category):
    '''计算输出与目标类别的损失，循环每一个样本，计算真实类别的logit值，值越大表示越接近真实类别
    Args:
        output: 网络输出，[batch,1000] tensor，未经过softmax
        target_category: 目标类别，list，长度为batch
    Returns:
        loss: 损失值，tensor
    '''
    loss = 0
    for i in range(len(target_category)):
        loss = loss + output[i, target_category[i]]
    return loss

def compute_logit_grad(model, X, y):
    '''计算logit层对应类别的神经元对输入的梯度'''
    model = model.to(device)
    model.eval()
    X = X.to(device)
    y = y.to(device)
    model.zero_grad()
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    output = model(X + delta)
    logit = get_loss(output, y)
    logit.backward()
    grad = delta.grad.detach().clone()
    return grad

def compute_output_grad(model, X, y):
    '''计算输出层（softmax之后）对输入的梯度'''
    model = model.to(device)
    model.eval()
    X = X.to(device)
    y = y.to(device)
    model.zero_grad()
    delta = torch.zeros_like(X, requires_grad=True).to(device)
    output = model(X + delta)
    output = nn.Softmax(dim=1)(output)
    softmax_loss = get_loss(output, y)
    softmax_loss.backward()
    grad = delta.grad.detach().clone()
    return grad

# -------------------- step2: 计算需要保留梯度的pixel --------------------
def grad_mask(grad, mode = None, **kwargs):
    '''对梯度进行掩码处理，生成于原始梯度相同形状的掩码，用于标记要修改的像素
    Args:mode
        'all': 返回全为1的掩码
        'positive': 梯度的正值为1，负值为0
        'negative': 梯度的负值为1，正值为0
        'topr':r为改变的pixel的比例，需要传入参数topr
        'randomr':随机选择randomr比例的像素，其余值置为0
        'channel_randomr':随机选择randomr比例的像素，但是3个通道最多只能有一个通道为1
    Return:
        mask: 掩码，[batch_size, channel, height, width], tensor, 1表示要修改的像素
        num_attacked: 要修改的像素数
    '''
    if mode == 'all' or mode is None:
        mask = torch.ones_like(grad)
        batchsize = mask.shape[0]
        num_attacked = mask.numel()//batchsize # 每张图片平均被修改的像素数
        return mask, num_attacked
    
    mode_dict = {
        'positive': grad_mask_positive,
        'negative': grad_mask_negative,
        'topr': grad_mask_topr,
        'lowr': grad_mask_lowr,
        'randomr': grad_mask_randomr,
        'channel_randomr': grad_mask_channel_randomr,
    }
    mask, num_attacked = mode_dict[mode](grad, **kwargs)
    return mask, num_attacked

def cam_mask(grayscale_cam, mode, **kwargs):
    '''对cam进行掩码处理，生成于原始cam相同形状的掩码，用于标记要修改的像素
    Args:mode
        'cam_topr':r为改变的pixel的比例，需要传入参数r
    '''
    
    mode_dict = {
        'cam_topr': cam_mask_topr,
        'cam_lowr': cam_mask_lowr,
    }
    mask, num_attacked = mode_dict[mode](grayscale_cam, **kwargs)
    return mask, num_attacked

def grad_mask_positive(grad):
    '''只保留梯度的正值，负值置为0,并返回梯度为正的pixel数'''
    positive_grad = torch.clamp(grad, min=0)
    num_positive = positive_grad[positive_grad > 0].numel()
    batchsize = positive_grad.shape[0]
    num_positive = int(num_positive // batchsize)
    mask = positive_grad.sign()
    return mask, num_positive

def grad_mask_negative(grad):
    '''只保留梯度的负值，正值置为0,并返回梯度为负的pixel数'''
    negative_grad = torch.clamp(grad, max=0)
    num_negative = negative_grad[negative_grad < 0].numel()
    # 除以batchsize,得到每个batch的平均负梯度数
    batchsize = negative_grad.shape[0]
    num_negative = int(num_negative // batchsize)
    mask = - negative_grad.sign()
    return mask, num_negative

def grad_mask_topr(grad, topr = 0.1):
    '''topr为改变的pixel的比例，梯度绝对值前topr比例的为1，其余为0'''
    grad = grad.abs()
    num_pixels = grad[0].numel()
    num_change_pixels = int(num_pixels * topr)
    top_array = compute_top_indics(grad, num_change_pixels)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels

def grad_mask_lowr(grad, lowr = 0.1):
    '''lowr为改变的pixel的比例，梯度绝对值后lowr比例的为1，其余为0'''
    grad = grad.abs()
    num_pixels = grad[0].numel()
    num_change_pixels = int(num_pixels * lowr)
    top_array = compute_top_indics(grad, num_change_pixels, ascending=True)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels

def grad_mask_randomr(grad, randomr=0.1):
    '''随机选择randomr比例的像素，其余值置为0,并返回被修改的pixel数'''
    batch_size = grad.shape[0]
    grad_view = grad.view(batch_size, -1)  
    mask = torch.zeros_like(grad_view)
    num_change_pixels = int(grad_view.shape[1] * randomr)
    for i in range(batch_size):
        rand_indices = torch.randperm(grad_view.shape[1])[:num_change_pixels]
        mask[i, rand_indices] = 1  # set the selected pixels to 1
    mask = mask.view_as(grad)
    return mask, num_change_pixels

def grad_mask_channel_randomr(grad, channel_randomr=0.1):
    '''随机选择randomr比例的像素，但是3个通道最多只能有一个通道为1'''
    batch_size, channels, height, width = grad.shape
    mask = torch.zeros_like(grad)
    num_pixels = height * width
    for i in range(batch_size):
        num_change_pixels = int(num_pixels * channel_randomr)
        rand_pixels = torch.randperm(num_pixels)[:num_change_pixels]
        for pixel in rand_pixels:
            rand_channel = torch.randint(0, channels, (1,))
            mask[i, rand_channel, pixel // width, pixel % width] = 1
    return mask, num_change_pixels

def cam_mask_topr(grayscale_cam, cam_topr = 0.1):
    '''topr为改变的pixel的比例，cam绝对值前topr比例的为1，其余为0，在三个通道中随机选一个通道
    Args:
        grayscale_cam: cam图, 形状为[batch_size, height, width]
        cam_topr: 改变的像素比例,小于1/3
    Returns:
        mask: 掩码，[batch_size, channel, height, width], tensor, 1表示要修改的像素
        num_change_pixels: 要修改的像素数
    '''
    cam = np.abs(grayscale_cam)
    num_pixels = cam[0].size # 224*224
    # 如果选择compute_top_indics使用的是distribute_values_to_channels不需要*3，否则需要
    num_change_pixels = int(num_pixels * cam_topr) 
    top_array = compute_top_indics(cam, num_change_pixels)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels*3

def cam_mask_lowr(grayscale_cam, cam_lowr = 0.1):
    '''lowr为改变的pixel的比例，cam绝对值后lowr比例的为1，其余为0，在三个通道中随机选一个通道
    Args:
        grayscale_cam: cam图, 形状为[batch_size, height, width]
        cam_lowr: 改变的像素比例,小于1/3
    Returns:
        mask: 掩码，[batch_size, channel, height, width], tensor, 1表示要修改的像素
        num_change_pixels: 要修改的像素数
    '''
    cam = np.abs(grayscale_cam)
    num_pixels = cam[0].size
    # 如果选择compute_top_indics使用的是distribute_values_to_channels不需要*3，否则需要
    num_change_pixels = int(num_pixels * cam_lowr) 
    top_array = compute_top_indics(cam, num_change_pixels, ascending=True)
    mask = torch.Tensor(top_array).to(device)
    return mask, num_change_pixels*3

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
        'GD': GD,
        'gaussian_noise': gaussian_noise,
        'gaussian_noise_sign': gaussian_noise_sign,
        'gaussian_noise_std': gaussian_noise_std,
        'gaussian_noise_sign_std': gaussian_noise_sign_std
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

def GD(eta_list, grad, **kwargs):
    '''GD方法'''
    perturbations = [torch.clamp(eta * grad, -eta, eta) for eta in eta_list]
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
    if fix:
        standard_perturb = torch.randn_like(grad) # 标准高斯噪声
        perturbations = [torch.clamp(eta * standard_perturb, -eta, eta) for eta in eta_list]
    else:
        perturbations = [torch.clamp(torch.randn_like(grad), -eta, eta) for eta in eta_list]
    return perturbations

def gaussian_noise_std(eta_list, grad, **kwargs):
    '''高斯噪声
    fix:固定白噪声的pattern
        fix=True:先生成(0,1)正态分布standard_preturb,然后对每个eta,
        用eta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
        fix=False:每个eta都随机生成一个（0，eta）的噪声
    '''
    torch.manual_seed(0) # 固定随机种子
    fix = kwargs.get('fix', False)
    target_std = 0.01  # 目标标准差
    
    if fix:
        standard_perturb = torch.randn_like(grad) * target_std  # 调整标准差
        perturbations = [torch.clamp(eta * standard_perturb, -eta, eta) for eta in eta_list]
    else:
        perturbations = [torch.clamp(torch.randn_like(grad) * target_std, -eta, eta) for eta in eta_list]
    
    return perturbations

def gaussian_noise_sign(eta_list, grad, **kwargs):
    '''高斯噪声
    fix:固定白噪声的pattern
        fix=True:先生成(0,1)正态分布standard_preturb,然后对每个eta,
        用eta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
        fix=False:每个eta都随机生成一个（0，eta）的噪声
    '''
    torch.manual_seed(0) # 固定随机种子
    fix = kwargs.get('fix', False)
    if fix:
        standard_perturb = torch.randn_like(grad) # 标准高斯噪声
        perturbations = [eta * standard_perturb for eta in eta_list]
    else:
        perturbations = [torch.randn_like(grad) * eta for eta in eta_list]
    
    # 根据grad的符号调整扰动方向
    adjusted_perturbations = []
    for perturb in perturbations:
        sign_grad = torch.sign(grad)
        adjusted_perturb = torch.where(sign_grad > 0, torch.abs(perturb), -torch.abs(perturb))
        adjusted_perturbations.append(adjusted_perturb)
    
    return adjusted_perturbations

def gaussian_noise_sign_std(eta_list, grad, **kwargs):
    '''高斯噪声
    fix:固定白噪声的pattern
        fix=True:先生成(0,1)正态分布standard_preturb,然后对每个eta,
        用eta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
        fix=False:每个eta都随机生成一个（0，eta）的噪声
    '''
    torch.manual_seed(0) # 固定随机种子
    fix = kwargs.get('fix', False)
    target_std = 0.01
    if fix:
        standard_perturb = torch.randn_like(grad) * target_std
        perturbations = [eta * standard_perturb for eta in eta_list]
    else:
        perturbations = [torch.clamp(torch.randn_like(grad) * target_std, -eta, eta) for eta in eta_list]
    
    # 根据grad的符号调整扰动方向
    adjusted_perturbations = []
    for perturb in perturbations:
        sign_grad = torch.sign(grad)
        adjusted_perturb = torch.where(sign_grad > 0, torch.abs(perturb), -torch.abs(perturb))
        adjusted_perturbations.append(adjusted_perturb)
    
    return adjusted_perturbations


    
# -------------------- step4: 生成对抗样本 --------------------
def generate_adv_images(images, perturbations):
    '''生成对抗样本
    Args:
        images: 原始图片, [batch_size, channel, height, width]
        perturbations: 扰动, [len(eta_list),batch_size, channel, height, width]
    '''
    if images.dim() == 4: # [batch_size, channel, height, width]
        images.unsqueeze(0) # [1, batch_size, channel, height, width]
    # images = (images - images.min()) / (images.max() - images.min())
    adv_images = [(images + perturbation).to(device) for perturbation in perturbations]
    return adv_images
   
if __name__ == '__main__':
    # model_str = 'vit_b_16'
    # model_str = 'resnet50'
    model_str = 'vgg16'
    model = load_model(model_str)
    data = torch.load('./data/images_100.pth')
    images, labels = data['images'], data['labels']
    save_path = './data/one_step_attack_try'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_classes = get_classes_with_index(labels)
    
    
    
        