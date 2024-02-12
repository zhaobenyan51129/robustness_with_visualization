'''grad-cam可视化方法的实现'''
import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from tools.get_classes import get_classes_with_pred, get_classes_with_index

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss
    
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads) # weights=目标layer的grad maps求平均[batch,channels,1,1]
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1) #目标layer的cam图像[batch,height,width]
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        # 目标layer的feature maps，list长为layer个数，每个元素shape为 [batch,channel,height,width]
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]  
        # 目标layer的grad maps [batch,channel,height,width]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]  
        target_size = self.get_target_width_height(input_tensor)  #（224,224）

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads) #（batch,height,width)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size) # 将cam图reshape到输入尺寸[batch,224,224]
            cam_per_target_layer.append(scaled[:, None, :])  #[batch,1,224,224]

        return cam_per_target_layer # 长度为layer的个数，每个元素形状为[batch,1,224,224]

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1) #[batch,layers,224,224]
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)  # 这里应该本来就没有负的，可能是考虑到精度
        result = np.mean(cam_per_target_layer, axis=1)  #对所有layer求均值 [batch,224,224]
        return self.scale_cam_image(result)  

    @staticmethod
    def scale_cam_image(cam, target_size=None): # cam:[batch,height,width]
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):
        '''计算grad-cam和梯度
        
        Args:
            input_tensor: 输入图像，[batch,3,224,224] tensor
            target_category: 目标类别，int或者list，长度为batch

        Returns:
            predicted_classes: 预测类别，list，长度为batch
            cam: grad-cam图像，[batch,224,224] numpy array
            # grad_of_input: 输入图像的梯度，[batch,224,224,3] numpy array
        
        '''
        if self.cuda:
            input_tensor = input_tensor.cuda()

        input_tensor.requires_grad_()  

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)  # [batch,1000]
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0) # input_tensor.size(0) = batch
            predicted_classes = get_classes_with_index(target_category)
            
        if isinstance(target_category, list):
            target_category = target_category
            predicted_classes = get_classes_with_index(target_category)

        if target_category is None:
            predicted_classes, target_category = get_classes_with_pred(output, 1)
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)  # 获取目标类别logit的值
        loss.backward(retain_graph=True)
        
        cam_per_layer = self.compute_cam_per_layer(input_tensor) # list,长度为layer个数，每个元素形状为[batch,1,224,224]
        cam = self.aggregate_multi_layers(cam_per_layer) # [batch,224,224]

        # 输出对输入的梯度
        # grad_of_input = input_tensor.grad.detach().cpu().numpy()  # Extract the gradient tensor
        # grad_of_input = grad_of_input.transpose(0, 2, 3, 1)  # [batch,224,224,3]
        
        # 交叉熵对输入的梯度
        # cross_entropy_loss = nn.CrossEntropyLoss()(output, torch.tensor(target_category).cuda())
        # cross_entropy_loss.backward()
        # grad = input_tensor.grad.detach().clone()
        return predicted_classes, cam

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray, # [batch,224,224,3]
                      mask: np.ndarray, # [batch,224,224]
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ 
    cv2.COLORMAP_JET 是一种预定义的颜色映射（colormap），用于将灰度图像转换为彩色图像
    cv2.cvtColor 函数将图像的颜色空间从 BGR（Blue-Green-Red）转换为 RGB（Red-Green-Blue）
    This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # 逐张图像处理并应用颜色映射
    heatmaps = []
    for gray_image in mask:
        heatmap = cv2.applyColorMap(np.uint8(255 * gray_image), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        heatmaps.append(heatmap)
    heatmap_np = np.stack(heatmaps)

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap_np + img

    # heatmap_np = heatmap_np / np.max(heatmap_np, axis=(1, 2), keepdims=True)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

def main():
    '''测试'''
    from data_preprocessor.load_images import load_images
    from data_preprocessor.normalize import apply_normalization
    from models.load_model import load_model
    from tools.show_images import show_images
    from visualization.reshape_tranform import ReshapeTransform

    model_str = 'vit_b_16'
    data_path = './select_images.pth'
    model = load_model(model_str)
    images, labels = load_images(data_path)
    # target_layers = [model.encoder.layers[-1].ln_1]
    target_layers = [model.blocks[-1].norm1]
    reshape_transform = ReshapeTransform(model)
    use_cuda = True
    cam = GradCAM(model=model, target_layers=target_layers,reshape_transform=reshape_transform, use_cuda=use_cuda)
    predicted_classes, grayscale_cam = cam(apply_normalization(images), target_category=None)
    img = images.permute(0, 2, 3, 1).cpu().numpy()
    vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    show_images(vis, predicted_classes, output_path='./data/grad_cam', save_name='grad_cam_vit_b16.jpg')

if __name__ == '__main__':
    main()