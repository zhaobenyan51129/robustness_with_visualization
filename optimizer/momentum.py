import torch


def attacker_mGD(delta, alpha, sign, momentum, decay, l1_norm, **kwargs):
    """
    data: [3, 224, 224] 经过反向传播的梯度
    """
    if sign == True:
        if l1_norm == True:
            momentum = decay * momentum + delta.grad / torch.norm(delta.grad, p=1)
        else:
            momentum = momentum * decay + (1-decay) * delta.grad.sign()
    else:
        momentum = momentum * decay + (1-decay) * delta.grad
    
    delta.data = delta.data - alpha * momentum