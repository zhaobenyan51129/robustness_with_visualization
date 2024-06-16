'''展示图片'''

import math
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import torch
import seaborn as sns

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
    num_rows = int(np.ceil(np.sqrt(batch_size)))
    num_cols = int(np.ceil(batch_size / num_rows))
    figsize = (num_cols * scale, (num_rows) * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16) 
    axes = axes.flatten()
    for i, (ax, image) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(image): # tensor
            image = image.cpu().numpy().transpose(1, 2, 0)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image - image.min()) / (image.max() - image.min())   
            image = np.clip(image, 0, 1)
        else:
            image = np.clip(image, 0, 255)
        ax.imshow(image)
        ax.axis("off")  
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_distribution(input, **kwargs): 
    '''显示分布

    Args:
        input: [batch, 3, 224, 224]， [batch, 224,224] numpy array or tensor
        kwargs: 包含以下可选参数的字典
            titles: 标题， 长度与batch相等的list
            output_path: 输出路径，如果不为None，则保存图片到指定路径
            save_name: 保存的图片名
            scale: 图片缩放比例，默认为2
            main_title: 图片的大标题
    '''
    titles = kwargs.get('titles', None)
    output_path = kwargs.get('output_path', None)
    save_name = kwargs.get('save_name', None)
    scale = kwargs.get('scale', 2)
    main_title = kwargs.get('main_title', None)

    batch_size = input.shape[0]
    num_rows = int(np.ceil(np.sqrt(batch_size)))
    num_cols = int(np.ceil(batch_size / num_rows))
    figsize = (num_cols * scale, (num_rows) * scale)
    space = 0.5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16) 
    axes = axes.flatten()
    for i, (ax, single_input) in enumerate(zip(axes, input)):
        if torch.is_tensor(single_input):# tensor
            single_input = single_input.cpu().numpy()
        flattened_single_input = single_input.flatten()
        sns.histplot(flattened_single_input, ax=ax, bins=50, kde=True, alpha=0.5)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2), useMathText=True)
        ax.xaxis.offsetText.set_fontsize(8)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
        ax.yaxis.offsetText.set_fontsize(8)
        ax.set_ylabel('')
       
    plt.subplots_adjust(hspace=space, wspace=space)
    
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_line_chart(x, y, output_path = None, save_name = None, title = None):
    '''画出y关于x的折线图
    Args:
        x: x轴的值
        y: y轴的值
        output_path: 输出路径，如果不为None，则保存图片到指定路径
        save_name: 保存的图片名
        title: 图片的大标题
    '''
    plt.plot(x, y)
    # for i, j in zip(x, y):
    #     plt.text(i, j, f'({round(i, 2)},{round(j, 2)})') 
    plt.title(title)
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from data_preprocessor.load_images import load_images
    from tools.get_classes import get_classes_with_index
    images, labels = load_images('./select_images.pth')
    classes = get_classes_with_index(labels)
    plot_distrubution(images, titles=classes, output_path='./data/show_images', save_name='distrubution.png')