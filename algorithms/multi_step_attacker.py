import torch
import torch.nn as nn
import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from tools.compute_topk import compute_top_indics

class AdversarialAttacksMultiStep(object):
    def __init__(self, model, X, y, alpha, eta, num_steps):
        '''
        Args:
            model: the model
            X: the original images, [batch_size, 3, 224, 224], tensor
            y: the labels of X, [batch_size,], tensor
            alpha: 扰动的步长
            eta: 扰动阈值
            num_steps: 迭代次数
        
        '''
        self.model = model
        self.X = X
        self.y = y
        self.alpha = alpha
        self.eta = eta
        self.num_steps = num_steps

    def i_fgsm(self, grad_mode = None, k = 10):
        '''Iterative Fast Gradient Sign Method'''
        self.delta = torch.zeros_like(self.X, requires_grad=True)
        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(self.model(self.X + self.delta), self.y)
            loss.backward()
           
            if grad_mode == 'i_fgsm_positive':
                grad = self.grad_seg_positive()
            elif grad_mode == 'i_fgsm_negative':
                grad = self.grad_seg_negative()
            elif grad_mode == f'i_fgsm_topk{k}':
                grad = self.grad_topk(k)
            else:
                grad = self.get_grad()
            self.delta.data = self.delta + self.alpha * grad.sign()
            self.delta.data = torch.clamp(self.delta, -self.eta, self.eta)
            self.delta.grad.zero_()

        self.grad = grad
        return self.delta.detach()

    def i_fgm(self, grad_mode = None, k=10):
        '''Iterative Fast Gradient Method'''
        self.delta = torch.zeros_like(self.X, requires_grad=True)
        batch_size = self.X.shape[0]
        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(self.model(self.X + self.delta), self.y)
            loss.backward()
            if grad_mode == 'i_fgm_positive':
                grad = self.grad_seg_positive()
            elif grad_mode == 'i_fgm_negative':   
                grad = self.grad_seg_negative()
            elif grad_mode == f'i_fgm_topk{k}':
                grad = self.grad_topk(k)
            else:
                grad = self.get_grad()
            normed_grad =  torch.norm(grad.view(batch_size, -1), p=2, dim=1)
            self.delta.data = self.delta + self.alpha * (grad / normed_grad.view(-1, 1, 1, 1))
            self.delta.data = torch.clamp(self.delta, -self.eta, self.eta)
            self.delta.grad.zero_()
        self.grad = grad
        return self.delta.detach()
    
    def pgd(self, grad_mode = None, k=10):
        '''Projected Gradient Descent'''
        self.delta = torch.zeros_like(self.X, requires_grad=True)
        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(self.model(self.X + self.delta), self.y)
            loss.backward()

            if grad_mode == 'pgd_positive':
                grad = self.grad_seg_positive()
            elif grad_mode == 'pgd_negative':   
                grad = self.grad_seg_negative()
            elif grad_mode == f'pgd_topk{k}':
                grad = self.grad_topk(k)
            else:
                grad = self.get_grad()

            self.delta.data = (self.delta + self.X.shape[0]*self.alpha*grad).clamp(-self.eta, self.eta)
            self.delta.grad.zero_()
        self.grad = grad
        return self.delta.detach()
    
    def i_gaussian_attck(self):
        '''Iterative Gaussian Attack'''
        self.delta = torch.zeros_like(self.X, requires_grad=False)
        for t in range(self.num_steps):
            perturbation = self.gaussian_noise()
            self.delta.data = self.delta + perturbation
        self.grad = None
        return self.delta
    
    def gaussian_noise(self):
        '''Gaussian Noise'''
        perturbation = torch.clamp(torch.randn_like(self.X), -self.eta, self.eta)
        return perturbation
    
    def grad_seg_positive(self):
        '''只保留梯度的正值，负数置为0'''
        gradient = self.get_grad()
        positive_grad = torch.clamp(gradient, min=0)
        return positive_grad
        
    def grad_seg_negative(self):
        '''只保留梯度的负值，正数置为0'''
        gradient = self.get_grad()
        negative_grad = torch.clamp(gradient, max=0)
        return negative_grad

    def grad_topk(self, k = 10):
        '''只保留梯度的前k个最大值，其余置为0'''
        gradient = self.get_grad()  
        # abs_grad = torch.abs(gradient)
        grad_np = gradient.cpu().numpy().transpose(0, 2, 3, 1)
        top_array, top_indices = compute_top_indics(grad_np, top_num=k)
        top_tensor = torch.from_numpy(top_array).cuda()
        topk_grad = torch.mul(top_tensor, gradient)
        return topk_grad
    
    def get_grad(self):
        '''获取梯度'''
        gradient = self.delta.grad.detach().clone()
        return gradient 
    
    def normalized(self, input_tensor):
        '''归一化'''
        max_value, min_value = input_tensor.max(), input_tensor.min()
        normalized_input_tensor = (input_tensor - min_value) / (max_value - min_value)
        return normalized_input_tensor
    