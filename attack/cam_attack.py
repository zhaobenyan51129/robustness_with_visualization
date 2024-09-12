import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
import argparse
import os
import shutil
import numpy as np
import torch
from data_preprocessor.load_images import load_images
from data_preprocessor.normalize import apply_normalization
from models.load_model import load_model
from tools.show_images import show_images, plot_distribution, plot_line_chart
from visualization.grad_cam import GradCAM, show_cam_on_image
from visualization.reshape_tranform import ReshapeTransform
from tqdm import tqdm
from tools.compute_topk import compute_top_indics

# 计算预测类别对于输入的梯度以及grad-cam图
class GradCAMWrapper:
    '''将GradCAM封装成一个类，方便调用'''
    def __init__(self, model_name = 'VGG16'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.model_name = model_name
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
        '''计算预测类别对于输入的梯度以及grad-cam图
        Args:
            input_tensor: 用于模型输入的tensor, 没有normalize, tensor[batch,3,224,224]
            target_category: 目标类别的索引，如果为None，则会用预测类别的索引
        '''
        predicted_classes, grayscale_cam= self.cam(input_tensor=apply_normalization(input_tensor), target_category=target_category)
        return predicted_classes, grayscale_cam
    
    def show_cam(self, img, grayscale_cam, labels, output_path = None, save_name = None, main_title = None):
        '''将grad-cam图叠加到原始图片上，并显示
        Args:
            img: 原始图片: numpy[batch, 224, 224, 3]
            grayscale_cam: grad-cam图，shape: [batch, 224, 224], numpy
            labels: 画图显示的label, 一般是预测类别，shape: [batch,]
            output_path: 图片保存路径
            save_name: 图片保存名称
            main_title: 图片的大标题
        '''
        img = self.__normalize(img)
        visualization = show_cam_on_image(img,
                                grayscale_cam,
                                use_rgb=True)
        if output_path:
            save_path = os.path.join(output_path, 'cam')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        show_images(visualization, labels, output_path=save_path, save_name = save_name, main_title = main_title)
    
    def __normalize(self, input_tensor):
        '''将输入的tensor归一化到[0,1]之间'''
        # norm = (input_tensor - input_tensor.mean())/input_tensor.std()
        # norm = norm * 0.1
        # norm = norm + 0.5
        # norm = norm.clip(0, 1)
        max_value, min_value = input_tensor.max(), input_tensor.min()
        normalized_input_tensor = (input_tensor - min_value) / (max_value - min_value)
        return normalized_input_tensor

class CamAttack:
    def __init__(self, gradcam, input_tensor, num = 200, ratio = 0.5):
        '''进行攻击的类
        Args:
            gradcam: 一个GradCAMWrapper对象，用于计算每一步的梯度和grad-cam图
            input_tensor: 用于模型输入的tensor, 没有normalize, tensor[batch, 3, 224, 224]
            num: 攻击的像素个数
            ratio: 攻击的强度，即添加的噪声的标准差
        '''
        self.gradcam = gradcam
        self.input_tensor = input_tensor
        self.num = num
        self.ratio = ratio
    
    def add_grey_to_channel(self, top_array, input):
        """将灰度图像的像素值加到输入张量的随机指定通道上。

        Args:
            top_array: 形状为 (batch_size, height, width) ,每个位置为0或1
            input: 形状为 (batch_size, num_channels, height, width) 的输入张量。

        Returns:
            形状为 (batch_size, num_channels, height, width) 的新张量。
        """
        num_channels = input.shape[1]
        batch_size = input.shape[0]
        channel_idx = torch.randint(0, num_channels, (batch_size,))
        attacked_tensor = input.clone()
        for i in range(input.shape[0]):
            noise = np.random.normal(loc=0.0, scale= self.ratio, size = top_array[i, :, :].shape)
            attacked_tensor[i, channel_idx[i], :, :] = input.cpu()[i, channel_idx[i], :, :] + top_array[i, :, :] * noise
        assert attacked_tensor.shape == input.shape

        return attacked_tensor
    
    def get_attacked_tensor(self, input, grayscale_cam = None, noise = None):
        '''input被攻击后的图片，可直接作为模型输入
        
        Args:
            input: 输入张量，shape: [batch, 3, 224, 224]
            grayscale_cam: cam特征图，shape: [batch, 224, 224]
            noise: 噪声
        '''
        if noise is None:
            noise = np.random.normal(loc=0.0, scale= self.ratio, size = input.shape)

        if grayscale_cam is not None:
            top_array = compute_top_indics(grayscale_cam, self.num)
            attacked_tensor = self.add_grey_to_channel(top_array, input).float()
        else: # 随机攻击
            mask = np.zeros(input.shape)
            one_indices = np.random.choice(np.prod(input.shape), size=self.num, replace=False)
            indices = np.unravel_index(one_indices, input.shape)
            mask[indices] = 1
            attacked_tensor = (input + mask * noise).float()

        return attacked_tensor

    def __call__(self, mode = 'random', target_category = None, max_loop = 1000, output_path = None):
        '''根据cam进行攻击
        Args:
            mode: 攻击模式，random或者cam
            target_category: 目标类别的索引，如果为None，则会用预测类别的索引
            max_loop: 最大迭代次数
            output_path: 输出路径
        '''
        if os.path.exists(output_path) and os.path.isdir(output_path):
            print(f'删除文件夹：{output_path}')
            shutil.rmtree(output_path)

        num_differences_list = []

        for i in range(max_loop + 1):
            loop_count = i 
            print(f'loop_count = {loop_count}')
            if i == 0:
                attacked_tensor = self.input_tensor
                original_classes, grayscale_cam = self.gradcam(attacked_tensor, target_category) 
                print(f"原始的预测类别 = {original_classes}")  
                predicted_classes = original_classes
            else: 
                if mode == 'random':
                    attacked_tensor = self.get_attacked_tensor(attacked_tensor)
                else:  
                    attacked_tensor = self.get_attacked_tensor(attacked_tensor, grayscale_cam)

                predicted_classes, grayscale_cam = self.gradcam(attacked_tensor, target_category)

            # title = [f'{orig}/{pred}' for orig, pred in zip(original_classes, predicted_classes)]
            title = [f'{orig}/{pred}' if orig != pred else f'{orig}' for orig, pred in zip(original_classes, predicted_classes)]
            img = attacked_tensor.permute(0, 2, 3, 1).cpu().numpy()

            num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(predicted_classes, original_classes))
            num_differences_list.append(num_differences)
            print(f'num_differences = {num_differences}')

            show_images(attacked_tensor, titles = title, output_path = os.path.join(output_path, 'attacked_images'), save_name = f'{str(loop_count)}.png', main_title = f'{num_differences} images are misclassified')
            self.gradcam.show_cam(img, grayscale_cam, title, output_path = output_path, save_name = f'{str(loop_count)}.png', main_title = f'{num_differences} images are misclassified')

            plot_distribution(grayscale_cam, titles = title, output_path = os.path.join(output_path, 'cam_distribution'), save_name = f'{str(loop_count)}.png', main_title = f'{num_differences} images are misclassified')

            if num_differences >= len(predicted_classes): # 所有的图片都被攻击成功
                print(f"攻击之后的预测类别：{predicted_classes}")
                result_dict = {'loop_count': loop_count, 
                        'num_differences_list': num_differences_list, 
                        'original_classes': original_classes, 
                        'predicted_classes': predicted_classes}
                return result_dict

        result_dict = {'loop_count': loop_count, 
                        'num_differences_list': num_differences_list, 
                        'original_classes': original_classes, 
                        'predicted_classes': predicted_classes}
        
        return result_dict

def parse_args():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser(description='Conduct attack experiments')
    parser.add_argument('--model_name', type=str, default='vit_b_16', help='model name')
    parser.add_argument('--ratio', type=float, default=0.5, help='ratio for attack')
    parser.add_argument('--attacked_pixel', type=int, default=200, help='number of attacked pixels')
    parser.add_argument('--max_loop', type=int, default=1000, help='max loop for attack')
    parser.add_argument('--output_path', type=str, default=None, help='output path for results')
    return parser.parse_args()

def main_multi_step():
    '''测试'''
    args = parse_args()
    model_name = args.model_name
    ratio = args.ratio
    attacked_pixel = args.attacked_pixel
    max_loop = args.max_loop
    mode = 'cam'

    output_path = f'./data/attacked_cam_multistep/{mode}/{attacked_pixel}_{ratio}'
    # input_tensor, labels = load_images(file_path=f'./select_images.pth')
    input_tensor, labels = load_images(file_path=f'./selected_images/data_100.pth')
    input_tensor = input_tensor[:36]
    labels = labels[:36]
    gradcam = GradCAMWrapper(model_name)
    attacker = CamAttack(gradcam, input_tensor, num=attacked_pixel, ratio = ratio)
    result_dict = attacker(mode=mode, max_loop=max_loop, output_path = output_path)

def main_onestep_attack():
    '''测试'''
    args = parse_args()
    model_name = args.model_name
    attacked_pixel = args.attacked_pixel
    max_loop = args.max_loop
    ratios = np.arange(0, 1, 0.1)
    succeed = []
    gradcam = GradCAMWrapper(model_name)
    mode = 'cam'
    for ratio in ratios:
        output_path = f'./data/attacked_cam_onestep/{mode}/{attacked_pixel}_{ratio}'
        input_tensor, labels = load_images(file_path=f'./selected_images/data_100.pth')
        input_tensor = input_tensor[:36]
        labels = labels[:36]
        attacker = CamAttack(gradcam, input_tensor, num=attacked_pixel, ratio = ratio)
        result_dict = attacker(mode=mode, max_loop=max_loop, output_path = output_path)
        # torch.save(result_dict, os.path.join(output_path, f'result_dict_{ratio}.pth'))
        changed_images = result_dict['num_differences_list'][-1]
        succeed.append(changed_images)

    print(succeed)
    save_dict = {'ratios': ratios, 'succeed': succeed}
    torch.save(save_dict, os.path.join(f'./data/attacked_cam_onestep/{mode}', f'save_dict.pth'))

    plot_line_chart(ratios, succeed, output_path = f'./data/attacked_cam_onestep/{mode}', save_name = f'random_succeed.png', title = 'num_images_succeed')


if __name__ == '__main__':
    # main_multi_step()
    # # 调用方式：python attack/cam_attack.py --model_name vit_b_16 --ratio 0.5 --attacked_pixel 2000 --max_loop 1000

    main_onestep_attack()
    # # 调用方式：python attack/cam_attack.py --model_name vit_b_16 --attacked_pixel 2000 --max_loop 1  
