import numpy as np
import torch
import torch.nn as nn
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from tools.compute_topk import compute_top_indics
from tools.get_classes import get_classes_with_index
from tools.show_images import show_images
from models.load_model import load_model
from data_preprocessor.normalize import apply_normalization
from visualization.grad_cam import GradCAM, show_cam_on_image


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'device: {device}')





