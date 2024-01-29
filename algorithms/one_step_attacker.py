import torch
import torch.nn as nn
import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from tools.show_images import show_images

class AdversarialAttacksOneStep(object):
    def __init__(self, model, X, y, eta):
        '''
        Args:
            model: the model
            X: the original images, [batch_size, 3, 224, 224], tensor
            y: the labels of X, [batch_size,], tensor
            eta: the eta (input variation parameter)'''
        self.model = model
        self.X = X
        self.eta = eta
        self.delta = torch.zeros_like(X, requires_grad=True)
        loss = nn.CrossEntropyLoss()(self.model(X + self.delta), y)
        loss.backward()

    def fgsm(self, grad = None):
        if grad is None:
            self.grad = self.get_grad()
        else:
            self.grad = grad
        perturbation = self.eta * self.grad.sign()
        return perturbation

    def fgm(self, grad = None):
        if grad is None:
            self.grad = self.get_grad()
        else:
            self.grad = grad
        norm_grad = torch.norm(self.grad, p=2)
        perturbation = self.eta * (self.grad / norm_grad)
        return perturbation
    
    def gaussian_noise(self):
        perturbation = torch.clamp(torch.randn_like(self.X), -self.eta, self.eta)
        return perturbation
    
    def grad_seg_positive(self):
        gradient = self.get_grad()
        positive_grad = torch.clamp(gradient, min=0)
        return positive_grad
        
    def grad_seg_negative(self):
        gradient = self.get_grad()
        negative_grad = torch.clamp(gradient, max=0)
        return negative_grad

    def grad_topk(self, k = 10):
        gradient = self.get_grad()  
        abs_grad = torch.abs(gradient)
        _, top_indices = torch.topk(abs_grad.view(self.grad.size(0), -1), k, dim=1, largest=True)
        topn_grad = torch.zeros_like(gradient)
        batch_indices = torch.arange(gradient.size(0)).view(-1, 1, 1, 1)
        topn_grad[batch_indices, :, top_indices // (224 * 224), top_indices % (224 * 224)] = gradient[batch_indices, :, top_indices // (224 * 224), top_indices % (224 * 224)]
        return topn_grad
    
    def get_grad(self):
        gradient = self.delta.grad.detach().clone()
        return gradient 
    
    def normalized(self, input_tensor):
        max_value, min_value = input_tensor.max(), input_tensor.min()
        normalized_input_tensor = (input_tensor - min_value) / (max_value - min_value)
        return normalized_input_tensor