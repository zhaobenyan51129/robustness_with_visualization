import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
import argparse
import os
import shutil
import numpy as np
import torch
from datasets.load_images import load_images
from datasets.normalize import apply_normalization
from models.load_model import load_model
from tools.show_images import show_images
from visualization.grad_cam import GradCAM, show_cam_on_image
from visualization.reshape_tranform import ReshapeTransform

# 计算预测类别对于输入的梯度以及grad-cam图
class GradCAMWrapper:
    def __init__(self, model_name = 'VGG16', save_name = None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.model_name = model_name
        self.save_name = save_name
        self.model = load_model(self.model_name)
        if self.model_name == 'vit_b_16':
            self.target_layers = [self.model.blocks[-1].norm1]
            self.reshape_transform = ReshapeTransform(self.model)
        elif self.model_name == 'ResNet':
            self.target_layers = [self.model.layer4[2].conv3]
            self.reshape_transform = None
        else:
          self.target_layers = [self.model.features[-1]]
        
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers, reshape_transform=self.reshape_transform, use_cuda=self.use_cuda)
        
        print(f"self.use_cuda = {self.use_cuda}")
    
    def __call__(self, input_tensor, target_category=None):
        predicted_classes, grayscale_cam, grad_of_input = self.cam(input_tensor=apply_normalization(input_tensor), target_category=target_category)
        return predicted_classes, grayscale_cam, grad_of_input
    
    def show_cam(self, img, grayscale_cam, predicted_classes):
        visualization = show_cam_on_image(img,
                                grayscale_cam,
                                use_rgb=True)
        show_images(visualization, predicted_classes, output_path=f'./data/output_{self.model_name}/cam', save_name = self.save_name)
    
    def __normalize(self, image):
        norm = (image - image.mean())/image.std()
        norm = norm * 0.1
        norm = norm + 0.5
        norm = norm.clip(0, 1)
        return norm
    
    def show_grad(self, grad_of_input, predicted_classes):
        grad = self.__normalize(grad_of_input)
        show_images(grad, predicted_classes, output_path=f'./data/output_{self.model_name}/grad', save_name = self.save_name)

class Attack:
    def __init__(self, gradcam, input_tensor, num = 200, ratio = 0.5):
        '''
        gradcam: 一个GradCAMWrapper对象，用于计算每一步的梯度和grad-cam图
        input_tensor: 用于模型输入的tensor, 没有normalize, tensor[batch,3,224,224]
        num: 攻击的像素个数
        ratio: 攻击的强度，即添加的噪声的标准差
        '''
        self.gradcam = gradcam
        self.input_tensor = input_tensor
        self.num = num
        self.ratio = ratio

    def compute_top_indics(self):
        '''计算输入数组的前n大的值和位置

        return: 
            top_array: [batch, 3, 224,224]，前n大的值为1，其余为0
            coordinates: 前n大值的位置, shape: (batch, top_num, dim - 1)
        '''
        batch = self.feature_array.shape[0]
        dim = np.ndim(self.feature_array)
        top_array = np.zeros_like(self.feature_array, dtype=int)
        coordinates = np.zeros((batch, self.num, dim - 1), dtype=int)
        for i in range(batch):
            if dim == 3:
                flattened_image = self.feature_array[i].flatten()
                top_indices = np.argpartition(flattened_image, -self.num)[-self.num:]
                coordinates_tmp = np.column_stack(np.unravel_index(top_indices, self.feature_array.shape[1:]))
                coordinates[i] = coordinates_tmp
                top_array[i].flat[top_indices] = 1 
            elif dim == 4:
                flattened_image = self.feature_array[i].reshape(-1, 3)
                top_indices = np.argpartition(flattened_image, -self.num, axis=None)[-self.num:]
                coordinates_tmp = np.column_stack(np.unravel_index(top_indices, self.feature_array.shape[1:]))
                coordinates[i] = coordinates_tmp
                top_array[i].flat[top_indices] = 1  
            else:
                print('不支持的维度')
        if dim == 4:
            top_array = np.transpose(top_array, (0,3,1,2))
        return top_array, coordinates
    
    def add_grey_to_channel(self, top_array):
        """将灰度图像的像素值加到输入张量的随机指定通道上。

        Args:
            top_array: 形状为 (batch_size, height, width) ,每个位置为0或1
            input_tensor: 形状为 (batch_size, num_channels, height, width) 的输入张量。

        Returns:
            形状为 (batch_size, num_channels, height, width) 的新张量。
        """
        num_channels = self.input_tensor.shape[1]
        batch_size = self.input_tensor.shape[0]
        channel_idx = torch.randint(0, num_channels, (batch_size,))
        attacked_tensor = self.input_tensor.clone()
        for i in range(self.input_tensor.shape[0]):
            noise = np.random.normal(loc=0.0, scale= self.ratio, size = top_array[i, :, :].shape)
            attacked_tensor[i, channel_idx[i], :, :] = self.input_tensor.cpu()[i, channel_idx[i], :, :] + top_array[i, :, :] * noise
        assert attacked_tensor.shape == self.input_tensor.shape
        return attacked_tensor
    
    def get_attacked_input_tensor(self):
        '''攻击后的图片，可直接作为模型输入'''
        noise = np.random.normal(loc=0.0, scale= self.ratio, size = self.input_tensor.shape)
        if self.mask == 'cam' or self.mask == 'grad':
            dim = np.ndim(self.feature_array)
            top_array, _ = self.compute_top_indics()
            if dim == 3:
                attacked_tensor = self.add_grey_to_channel(top_array).float()
            elif dim == 4:
                attacked_tensor = (self.input_tensor + top_array * noise).float()
            else:
                print('不支持的维度')

        else: # self.mode == 'random':
            shape = self.input_tensor.shape
            mask = np.zeros(shape)
            one_indices = np.random.choice(np.prod(shape), size=self.num, replace=False)
            indices = np.unravel_index(one_indices, shape)
            mask[indices] = 1
            attacked_tensor = (self.input_tensor + mask * noise).float()
        assert attacked_tensor.shape == self.input_tensor.shape  
        return attacked_tensor
    
    def predict_and_attack(self, ratio_of_succeed = 0.5, mask = 'cam', max_loop = 1000, output_path = None):
        self.mask = mask
        if os.path.exists(output_path) and os.path.isdir(output_path):
            print(f'删除文件夹：{output_path}')
            shutil.rmtree(output_path)
        original_classes, grayscale_cam, grad_of_input = self.gradcam(self.input_tensor)

        loop_count = 0
        num_differences_list = []
        show_images(self.input_tensor, titles = original_classes, output_path = output_path, save_name = f'{str(loop_count)}.png')
        print(f"攻击方式 = {self.mask}")
        print(f"原始的预测类别 = {original_classes}")
        if self.mask == 'cam':
            self.feature_array = grayscale_cam
        elif self.mask == 'grad':
            self.feature_array = grad_of_input
        elif self.mask == 'random':
            pass
        else:
            raise ValueError("Invalid value for 'mask'. Use 'cam', 'grad', or 'random'.")

        self.attacked_tensor = self.get_attacked_input_tensor()
        num_differences_list.append(0)
        while loop_count <= max_loop:
            loop_count += 1
            print(f'loop_count = {loop_count}')
            predicted_classes, grayscale_cam, grad_of_input = self.gradcam(self.attacked_tensor)
            show_images(self.attacked_tensor, titles = predicted_classes, output_path = output_path, save_name = f'{str(loop_count)}.png')
            num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(predicted_classes, original_classes))
            num_differences_list.append(num_differences)
            if num_differences >= len(predicted_classes) * ratio_of_succeed:
                print(f"攻击之后的预测类别：{predicted_classes}")
                return loop_count, num_differences_list, original_classes, predicted_classes
            else:
                if self.mask == 'cam':
                    self.feature_array = grayscale_cam
                elif self.mask == 'grad':
                    self.feature_array = grad_of_input
                else:
                    pass
                self.input_tensor = self.attacked_tensor
                self.attacked_tensor = self.get_attacked_input_tensor()
        assert loop_count == max_loop + 1, "The loop should have been stopped by now."
        assert len(num_differences_list) == loop_count + 1
        return loop_count, num_differences_list, original_classes, predicted_classes

def parse_args():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser(description='Conduct attack experiments')
    parser.add_argument('image_input', type=str, help='path to input image')
    parser.add_argument('--model_name', type=str, default='vit_b_16', help='model name')
    parser.add_argument('--mask', type=str, default='cam', choices=['grad', 'cam', 'random'], help='mask type')
    parser.add_argument('--ratio', type=float, default=0.5, help='ratio for attack')
    parser.add_argument('--attacked_pixel', type=int, default=200, help='number of attacked pixels')
    parser.add_argument('--ratio_of_succeed', type=float, default=0.5, help='ratio of succeed for attack')
    parser.add_argument('--max_loop', type=int, default=1000, help='max loop for attack')
    parser.add_argument('--output_path', type=str, default=None, help='output path for results')
    return parser.parse_args()

def main():
    '''测试'''
    args = parse_args()
    image_input = args.image_input
    model_name = args.model_name
    mask = args.mask
    ratio = args.ratio
    attacked_pixel = args.attacked_pixel
    ratio_of_succeed = args.ratio_of_succeed
    max_loop = args.max_loop
    output_path = args.output_path if args.output_path else f'./data/{image_input}_attacked_{mask}/{attacked_pixel}_{ratio}_{ratio_of_succeed}'

    input_tensor, labels = load_images(file_path=f'./select_images.pth')
    gradcam = GradCAMWrapper(model_name)
    output_path = f'./data/{image_input}_attacked_{mask}/{ratio}_{ratio_of_succeed}'

    attacker = Attack(gradcam, input_tensor, num=attacked_pixel, ratio = ratio)
    loop_count, num_differences_list, original_classes, predicted_classes = attacker.predict_and_attack(ratio_of_succeed = ratio_of_succeed, mask = mask, max_loop=max_loop, output_path = output_path)

if __name__ == '__main__':
    main()
    # 调用方式：python attack/cam_grad_attack.py select_images --model_name vit_b_16 --mask cam --ratio 0.5 --attacked_pixel 2000 --ratio_of_succeed 0.5 --max_loop 1000 --output_path ./data/attacked_cam/2000_0.5_0.5
