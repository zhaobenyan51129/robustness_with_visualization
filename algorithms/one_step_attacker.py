import torch
import torch.nn as nn
import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from tools.compute_topk import compute_top_indics

class AdversarialAttacksOneStep(object):
    def __init__(self, model, X, y, eta):
        '''
        Args:
            model: the model
            X: the original images, [batch_size, 3, 224, 224], tensor
            y: the labels of X, [batch_size,], tensor
            eta: 扰动阈值
        '''
        self.model = model
        self.X = X
        self.eta = eta
        self.delta = torch.zeros_like(X, requires_grad=True)
        loss = nn.CrossEntropyLoss()(self.model(X + self.delta), y)
        loss.backward()

    def fgsm(self, grad = None):
        '''Fast Gradient Sign Method'''
        if grad is None:
            self.grad = self.get_grad()
        else:
            self.grad = grad
        perturbation = self.eta * self.grad.sign()
        return perturbation

    def fgm(self, grad = None):
        '''Fast Gradient Method'''
        if grad is None:
            self.grad = self.get_grad()
        else:
            self.grad = grad
        norm_grad = torch.norm(self.grad, p=2)
        perturbation = self.eta * (self.grad / norm_grad)
        return perturbation
    
    def gaussian_noise(self):
        '''Gaussian noise'''
        perturbation = torch.clamp(torch.randn_like(self.X), -self.eta, self.eta)
        return perturbation
    
    def grad_seg_positive(self):
        '''只保留梯度的正值，负值置为0'''
        gradient = self.get_grad()
        positive_grad = torch.clamp(gradient, min=0)
        return positive_grad
        
    def grad_seg_negative(self):
        '''只保留梯度的负值，正值置为0'''
        gradient = self.get_grad()
        negative_grad = torch.clamp(gradient, max=0)
        return negative_grad

    def grad_topk(self, k = 10):
        '''保留梯度的top k值，其余置为0'''
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
    