'''初始化一个优化器（optimizer）和待优化变量（delta）'''
import torch
import torch.optim as optim


def init_optim(image_tensor, optimizer_str, learning_rate):
    """ 
    image_tensor: [3, 224, 224]
    optimizet_str: 'SGD', 'mSGD', 'Adagrad','RMSprop','Adam'
    """
    image_tensor = image_tensor.cuda()
    delta = torch.zeros_like(image_tensor, requires_grad=True).cuda()

    if optimizer_str == 'mSGD':
        optimizer = getattr(optim, 'SGD')([delta], lr=learning_rate, momentum=0.9)
    # elif optimizer_str == 'RMSprop':
    #     optimizer = optim.RMSprop([delta], lr=learning_rate, alpha=0.999)
    else:
        optimizer = getattr(optim, optimizer_str)([delta], lr=learning_rate)
    
    return delta, optimizer