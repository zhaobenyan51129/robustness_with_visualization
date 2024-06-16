import numpy as np
import torch

def compute_top_indics(X, top_num=10):
    '''计算输入数组的前n大的值和位置
    Args:
        X: 输入数组，shape: (batch, 224, 224, 3) or (batch, 224,224), numpy
        top_num: 前n大的值
    return: 
        top_array: numpy[batch, 3, 224,224] or [batch, 224,224] ，前n大的值为1，其余为0
        coordinates: 前n大值的位置, numpy, shape: (batch, top_num, dim - 1)
    '''
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    
    batch = X.shape[0]
    dim = np.ndim(X)
    top_array = np.zeros_like(X, dtype=int)
    coordinates = np.zeros((batch, top_num, dim - 1), dtype=int)
    for i in range(batch):
        if dim == 3:
            flattened_image = X[i].flatten()
            top_indices = np.argpartition(flattened_image, -top_num)[-top_num:]
            coordinates_tmp = np.column_stack(np.unravel_index(top_indices, X.shape[1:]))
            coordinates[i] = coordinates_tmp
            top_array[i].flat[top_indices] = 1 
        elif dim == 4:
            flattened_image = X[i].reshape(-1, 3)
            top_indices = np.argpartition(flattened_image, -top_num, axis=None)[-top_num:]
            coordinates_tmp = np.column_stack(np.unravel_index(top_indices, X.shape[1:]))
            coordinates[i] = coordinates_tmp
            top_array[i].flat[top_indices] = 1  
        else:
            print('不支持的维度')
    return top_array, coordinates

# def compute_top_indics(X, top_num=10):
#     '''计算输入数组的前n大的值和位置
#     Args:
#         X: 输入数组，shape: (batch, 224, 224, 3) or (batch, 224,224), tensor or numpy array
#         top_num: 前n大的值
#     return: 
#         top_array: tensor[batch, 3, 224,224] or [batch, 224,224] ，前n大的值为1，其余为0
#         coordinates: 前n大值的位置, tensor, shape: (batch, top_num, dim - 1)
#     '''
#     if isinstance(X, np.ndarray):
#         X = torch.from_numpy(X)
#     elif isinstance(X, torch.Tensor):
#         X = X.detach().cpu()
#     else:
#         raise TypeError("X must be a numpy array or a PyTorch tensor.")
    
#     batch = X.shape[0]
#     dim = X.dim()
#     top_array = torch.zeros_like(X, dtype=torch.int)
#     coordinates = torch.zeros((batch, top_num, dim - 1), dtype=torch.int)
    
#     for i in range(batch):
#         flattened_image = X[i].view(-1)
#         top_indices = torch.topk(flattened_image, top_num)[1]
#         coordinates[i] = torch.stack(torch.unravel_index(top_indices, X.shape[1:])).T
#         top_array[i].view(-1)[top_indices] = 1 

#     if dim == 4:
#         top_array = top_array.permute(0,3,1,2)
    
#     return top_array, coordinates