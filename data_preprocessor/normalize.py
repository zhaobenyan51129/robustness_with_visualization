'''标准化操作'''
import torch.nn as nn
import torch
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def apply_normalization(imgs):
    """
    ImageNet图片喂入模型前的标准化处理
    
    Args:
        imgs: 图片数据, [b, 3, 224, 224], 数据类型为torch.float32
    """
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    normalize_layer = Normalize(mean, std)
    imgs_tensor = normalize_layer(imgs)
    return imgs_tensor