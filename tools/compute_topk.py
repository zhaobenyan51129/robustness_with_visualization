import numpy as np
import torch

def compute_top_indics(X, top_num=10, ascending=False):
    '''计算输入数组的前n大的值和位置
    Args:
        X: 输入数组，shape: (batch, 224, 224, 3) or (batch, 224,224), numpy
        top_num: 前n大的值
        ascending: 是否按升序排列, 默认为False，即降序排列，取前n大的值
    return: 
        top_array: numpy[batch, 3, 224,224] or [batch, 224,224] ，前n大的值为1，其余为0
        coordinates: 前n大值的位置, numpy, shape: (batch, top_num, dim - 1)(无需返回)
    '''
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    
    batch = X.shape[0]
    dim = np.ndim(X)
    top_array = np.zeros_like(X, dtype=int)
    # coordinates = np.zeros((batch, top_num, dim - 1), dtype=int)
    for i in range(batch):
        if dim == 3:
            flattened_image = X[i].flatten()
            if ascending:
                top_indices = np.argpartition(flattened_image, top_num)[:top_num] # 找到前 k 个最小的元素的索引
            else:
                top_indices = np.argpartition(flattened_image, -top_num)[-top_num:] # 找到前 k 个最大的元素的索引
            # coordinates_tmp = np.column_stack(np.unravel_index(top_indices, X.shape[1:]))
            # coordinates[i] = coordinates_tmp
            top_array[i].flat[top_indices] = 1 
        elif dim == 4:
            flattened_image = X[i].reshape(-1, 3)
            if ascending:
                top_indices = np.argpartition(flattened_image, top_num, axis=None)[:top_num]
            else:
                top_indices = np.argpartition(flattened_image, -top_num, axis=None)[-top_num:]
            # coordinates_tmp = np.column_stack(np.unravel_index(top_indices, X.shape[1:]))
            # coordinates[i] = coordinates_tmp
            top_array[i].flat[top_indices] = 1  
        else:
            print('不支持的维度')
    # top_array = distribute_values_to_channels_random(top_array)
    top_array = distribute_values_to_channels(top_array)
    return top_array #, coordinates

def distribute_values_to_channels(top_array):
    '''如果top_array的形状为[batch, 224, 224]，将top_array的值复制到3个通道中'''
    if len(top_array.shape) == 4:
        return top_array
    output_array = np.repeat(top_array[:, np.newaxis, :, :], 3, axis=1)
    return output_array

def distribute_values_to_channels_random(top_array):
    '''如果top_array的形状为[batch, 224, 224] 将top_array的值随机分配到3个通道中'''
    if len(top_array.shape) == 4:
        return top_array
    batch_size = top_array.shape[0]
    height = top_array.shape[1]
    width = top_array.shape[2]
    # 步骤1: 创建输出数组
    output_array = np.zeros((batch_size, 3, height, width), dtype=top_array.dtype)
    
    # 遍历每个批次和每个位置
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                # 随机选择一个通道
                channel = np.random.randint(0, 3)
                # 将原始数组的值赋给选中的通道
                output_array[b, channel, i, j] = top_array[b, i, j]
                
    return output_array


