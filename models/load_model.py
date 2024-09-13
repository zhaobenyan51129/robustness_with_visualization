'''加载模型'''
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, vit_b_32, vit_l_32, vit_b_16, vit_l_16, vgg16
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.vit_model import vit_base_patch16_224

def load_model(model_str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        map_location = "cuda"
    else:
        map_location = "cpu"
        
    if model_str == 'resnet34':
        model = resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
    elif model_str == 'resnet50':
        model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    elif model_str == 'resnet101':
        model = resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
    elif model_str == 'resnet152':
        model = resnet152(weights='ResNet152_Weights.IMAGENET1K_V1')
    elif model_str == 'vit_b_32':
        model = vit_b_32(weights='ViT_B_32_Weights.IMAGENET1K_V1')
    elif model_str == 'vit_l_32':
        model = vit_l_32(weights='ViT_L_32_Weights.IMAGENET1K_V1')
    elif model_str == 'vit_b_16':
        model = vit_base_patch16_224()
        weights_path = "./data/vit_base_patch16_224.pth"
        model.load_state_dict(torch.load(weights_path, map_location = map_location))
       
        # model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
    elif model_str == 'vit_l_16':
        model = vit_l_16(weights='ViT_L_16_Weights.IMAGENET1K_V1') 
    elif model_str == 'vgg16':
        model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    else:
        raise Exception('model_str error!')
    model.eval()
    model.to(device)
    if torch.cuda.device_count() > 1:
        # print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    return model