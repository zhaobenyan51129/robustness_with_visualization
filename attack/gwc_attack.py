# We implement the non-targeted attack in this file.
import torch
import torch.nn as nn
import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')

from models.load_model import load_model
from datasets.load_images import load_images
from datasets.normalize import apply_normalization
import optimizer.learning_rate_rule as lr_rules
from tqdm import tqdm

def is_save(epoch):
    if epoch <= 500:
        return True
    else:
        if epoch % 10 == 0:
            return True
        else:
            return False

def single_attack(model, image, label, epsilon, num_epoch, lr, lr_rule, attacker, is_sign):
    """ 
    对一张图，按照指定优化器，epsilon与学习率，进行非目标对抗攻击
    image_tensor: [3, 224, 224]
    label: int
    epsilon: float
    lr: float

    loss_tensor: [epoch,]
    delta_before_clipping_tensor: [epoch(is_save), 3, 224, 224]
    gradient_tensor: [epoch(is_save), 3, 224, 224]
    epoch_tensor: [epoch(is_save),]
    """

    model.zero_grad()
    image = image.cuda()
    delta = torch.zeros_like(image, requires_grad=True).cuda()

    loss_list = [] # 记录损失
    delta_before_clipping_list = [] # clipping之前的perturbation
    gradient_list = [] # 记录gradient
    epoch_list = []

    momentum = torch.zeros_like(image).detach().cuda()
    progress_bar = tqdm(total=100, desc='Training', position=0, leave=True)

    for t in range(num_epoch):

        if t >= 10000:
            break
        is_save_bool = is_save(t)

        ### 梯度清零
        if t > 0:
            delta.grad.zero_()
        model.zero_grad()

        ### 前向传播
        pred = model(apply_normalization(image + delta)) # 前向传播
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([label]).cuda()) # 计算损失
        if t % (num_epoch // 100) == 0:
            progress_bar.set_postfix(loss=loss, refresh=True)
            progress_bar.update(1)
        loss_list.append(loss.item()) # 记录损失
        loss_tensor = torch.tensor(loss_list)

        ### stopping criteria
        if t > 500:
            if  (loss_tensor[:-200].min() - loss_tensor.min())/ abs(loss_tensor.min()) <= 1e-2:
                loss_list.pop()
                break

        if is_save_bool:
            epoch_list.append(t)

        # if t % 5 == 0:
        #     print(t, 'loss：', loss.item())

        ### 反向传播并记录梯度
        loss.backward() # 反向传播
        if is_save_bool:
            gradient_list.append(delta.grad.detach().clone()) # 存梯度ss

        if lr_rule:
            alpha = lr * lr_rule(t)
        else:
            alpha=lr
            
        attacker(delta, alpha, sign=is_sign) ###

        if is_save_bool:
            delta_before_clipping_list.append(delta.data.detach().clone()) # 记录perturbation

        delta.data.clamp_(-epsilon, epsilon)# projection
    progress_bar.close()
    
    loss_tensor = torch.tensor(loss_list)
    delta_before_clipping_tensor = (torch.stack(delta_before_clipping_list)).detach().cpu()
    gradient_tensor = (torch.stack(gradient_list)).detach().cpu()
    epoch_tensor = torch.tensor(epoch_list)

    return loss_tensor, delta_before_clipping_tensor, gradient_tensor, epoch_tensor

def attacker_GD(delta, alpha, sign=False, **kwargs):
    """
    data: [3, 224, 224] 刚经过反向传播的梯度
    """
    if sign == True:
        delta.data = delta.data - alpha * delta.grad.sign() # FGSM
    else:
        delta.data = delta.data - alpha * delta.grad # FGM
    
def main():
    images, labels = load_images('./select_images.pth')
    image = images[0]
    label = labels[0]
    model = load_model('resnet50')
    epsilon = 0.01
    num_epoch = 10000
    lr = 1
    lr_rule = lr_rules.big_start_exp_lr
    attacker = attacker_GD
    is_sign = False
    loss_tensor, delta_before_clipping_tensor, gradient_tensor, epoch_tensor =  single_attack(model, image, label, epsilon, num_epoch, lr, lr_rule, attacker, is_sign)

if __name__ == '__main__':
    main()