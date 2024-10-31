'''展示图片'''
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns

# import matplotlib
# matplotlib.rc("font", family='DejaVu Sans Mono')

def show_images(imgs, **kwargs):
    '''显示图片

    Args:
        imgs: (batch,224,224,3) numpy array, or [batch,3,224,224] tensor
        kwargs: 包含以下可选参数的字典
            titles: 标题， 长度与batch相等的list
            output_path: 输出路径，如果不为None，则保存图片到指定路径
            save_name: 保存的图片名
            scale: 图片缩放比例，默认为1.5
            main_title: 图片的大标题
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    scale = kwargs.get('scale', 1.5)
    main_title = kwargs.get('main_title', None)

    batch_size = imgs.shape[0]
    num_rows = kwargs.get('nrows', int(np.ceil(np.sqrt(batch_size))))
    num_cols = kwargs.get('ncols', int(np.ceil(batch_size / num_rows)))
    
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16) 

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (ax, image) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(image): # tensor
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min) # 归一化到0-1，否则imshow会报错
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image * 255
        ax.imshow(image)
        ax.axis("off")  
        if titles is not None:
            if '/' in titles[i]:
                ax.set_title(titles[i], fontsize=8, color='red')
            else:
                ax.set_title(titles[i], fontsize=8)
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, save_name))
        plt.close()
    else:
        plt.show()
    

def show_pixel_distribution(imgs, **kwargs):
    '''显示像素分布

    Args:
        imgs: (batch,224,224,3) numpy array, or [batch,3,224,224] tensor
        kwargs: 包含以下可选参数的字典
            titles: 标题， 长度与batch相等的list
            output_path: 输出路径，如果不为None，则保存图片到指定路径
            save_name: 保存的图片名
            scale: 图片缩放比例，默认为1.5
            main_title: 图片的大标题
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    scale = kwargs.get('scale', 1.5)
    main_title = kwargs.get('main_title', None)
    
    # 确保 imgs 是 numpy 数组，并具有批次维度
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().detach().numpy()
    elif not isinstance(imgs, np.ndarray):
        imgs = np.array(imgs)

    if imgs.ndim == 3:
        imgs = imgs[np.newaxis, ...]  # 添加批次维度

    batch_size = imgs.shape[0]
    num_rows = kwargs.get('nrows', int(np.ceil(np.sqrt(batch_size))))
    num_cols = kwargs.get('ncols', int(np.ceil(batch_size / num_rows)))
    # num_rows = kwargs.get('nrows', None)
    # num_cols = kwargs.get('ncols', None)

    # batch_size = imgs.shape[0]
    # if num_rows is None:
    #     num_rows = int(np.ceil(np.sqrt(batch_size)))
    # if num_cols is None:
    #     num_cols = int(np.ceil(batch_size / num_rows))
    
    figsize = (num_cols * scale, num_rows * scale)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    if main_title:
        fig.suptitle(main_title, fontsize=16) 

    # 确保 axs 是一维可迭代对象
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    elif isinstance(axs, (list, tuple)):
        pass  # axs 已经是可迭代的
    else:
        axs = [axs]  # 将单个 Axes 对象放入列表

    for i, ax in enumerate(axs):
        if i < len(imgs):  # 确保索引不超出范围
            image = imgs[i]
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if image.ndim == 3:  # 如果是 (H, W, C)，展开为一维
                image = image.flatten()
            elif image.ndim == 4:  # 如果是 (C, H, W)，转换为 (H, W, C)
                image = image.transpose(1, 2, 0).flatten()
            else:
                image = image.flatten()
            ax.hist(image, bins=100)
            if titles is not None and len(titles) > i:
                if '/' in titles[i]:
                    ax.set_title(titles[i], fontsize=6, color='red')
                else:
                    ax.set_title(titles[i], fontsize=6)
        else:
            ax.axis('off')  # 隐藏多余的子图
    plt.tight_layout()
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
 
def show_gradient_distribution(gradients, **kwargs):
    '''Display gradient distributions.

    Args:
        gradients: Tensor of gradients, shape (batch_size, channels, height, width)
        kwargs: Contains the following optional parameters:
            titles: List of titles, length equal to batch_size
            output_path: Output path, if not None, save the image to the specified path
            save_name: Name of the saved image
            scale: Image scaling factor, default is 1.5
            main_title: Main title of the image
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', 'gradient_distributions.png')
    scale = kwargs.get('scale', 1.5)
    main_title = kwargs.get('main_title', 'Gradient Distributions')
    
    batch_size = gradients.shape[0]
    num_rows = kwargs.get('nrows', int(np.ceil(np.sqrt(batch_size))))
    num_cols = kwargs.get('ncols', int(np.ceil(batch_size / num_rows)))
    figsize = (num_cols * scale, num_rows * scale)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16)
    
    # Flatten axs for easy iteration
    if num_rows * num_cols == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i < batch_size:
            grad_i = gradients[i]  # Shape: (channels, height, width)
            grad_i_flat = grad_i.view(-1).cpu().detach().numpy()
            ax.hist(grad_i_flat, bins=100, color='blue', alpha=0.7)
            if titles and i < len(titles):
                if '/' in titles[i]:
                    ax.set_title(titles[i], fontsize=8, color='red')
                else:
                    ax.set_title(titles[i], fontsize=8)
        else:
            ax.axis('off')  # Hide extra subplots

    plt.tight_layout()
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, save_name)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
    
def visualize_masks_overlay(images, masks, **kwargs):
    '''
    Overlay masks on the original images.

    Args:
        images: Tensor of images, shape (batch_size, channels, height, width)
        masks: Tensor of masks, same shape as images
        kwargs: Optional parameters
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', 'mask_overlay_visualization.png')
    scale = kwargs.get('scale', 1.5)
    main_title = kwargs.get('main_title', None)

    batch_size = images.shape[0]

    num_rows = kwargs.get('nrows', int(np.ceil(np.sqrt(batch_size))))
    num_cols = kwargs.get('ncols', int(np.ceil(batch_size / num_rows)))

    figsize = (num_cols * scale, num_rows * scale)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16)

    if num_rows * num_cols == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < batch_size:
            image_i = images[i].cpu().detach().numpy().transpose(1, 2, 0)
            mask_i = masks[i].max(dim=0)[0].cpu().detach().numpy()
            # Normalize image to [0, 1] for visualization
            image_i = (image_i - image_i.min()) / (image_i.max() - image_i.min())
            # Create an overlay by adding the mask to the image
            overlay = image_i.copy()
            overlay[mask_i > 0] = [1, 0, 0]  # Highlight mask regions in red
            ax.imshow(overlay)
            ax.axis('off')
            if titles is not None:
                if '/' in titles[i]:
                    ax.set_title(titles[i], fontsize=8, color='red')
                else:
                    ax.set_title(titles[i], fontsize=8)
        else:
            ax.axis('off')

    plt.tight_layout()
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, save_name)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()


def visualize_gradients(gradients, **kwargs):
    '''
    可视化梯度的热力图。

    Args:
        gradients: [batch, 3, H, W] 的张量，表示梯度。
        kwargs: 可选参数，包括：
            - titles: 每个图像的标题列表。
            - output_path: 保存输出图像的路径。
            - save_name: 保存的图像文件名。
            - scale: 图像大小的缩放因子。
            - main_title: 图像的主标题。
            - cmap: 热力图的颜色映射，默认为 'viridis'。
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', 'gradient_heatmaps.png')
    scale = kwargs.get('scale', 1.5)
    main_title = kwargs.get('main_title', None)
    cmap = kwargs.get('cmap', 'viridis')

    # 将梯度转换为 numpy 数组
    if isinstance(gradients, torch.Tensor):
        gradients = gradients.detach().cpu().numpy()

    batch_size = gradients.shape[0]
    num_rows = kwargs.get('nrows', int(np.ceil(np.sqrt(batch_size))))
    num_cols = kwargs.get('ncols', int(np.ceil(batch_size / num_rows)))

    figsize = (num_cols * scale, num_rows * scale)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)

    if main_title:
        fig.suptitle(main_title, fontsize=16)

    # 确保 axs 是一维数组
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = np.array([axs])  # 将单个 Axes 对象转换为数组

    for idx in range(batch_size):
        grad = gradients[idx]
        grad_abs = np.abs(grad)
        # 按通道求和，得到单通道梯度
        grad_sum = np.sum(grad_abs, axis=0)  # 形状：[H, W]
        grad_norm = grad_sum / (grad_sum.max() + 1e-8)
        ax = axs[idx]
        im = ax.imshow(grad_norm, cmap=cmap)
        ax.axis('off')
        if titles is not None:
            if '/' in titles[idx]:
                ax.set_title(titles[idx], fontsize=8, color='red')
            else:
                ax.set_title(titles[idx], fontsize=8)

    for idx in range(batch_size, len(axs)):
        axs[idx].axis('off')

    # 添加颜色条
    # fig.colorbar(im, ax=axs[:batch_size], orientation='vertical', fraction=0.02, pad=0.04)

    plt.tight_layout()
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, save_name), dpi=300)
        plt.close()
    else:
        plt.show()




