'''展示图片'''

import os
from matplotlib import pyplot as plt
import numpy as np
import torch

def show_images(imgs, titles = None, output_path = None, save_name = None, scale=1.5, main_title = None): 
    '''显示图片

    Args:
        imgs: (batch,224,224,3) numpy array, or [batch,3,224,224] tensor
        titles: 标题， 长度与batch相等的list
        output_path: 输出路径，如果不为None，则保存图片到指定路径
        save_name: 保存的图片名
        scale: 图片缩放比例
        main_title: 图片的大标题
    '''
    batch_size = imgs.shape[0]
    num_rows = int(np.ceil(np.sqrt(batch_size)))
    num_cols = int(np.ceil(batch_size / num_rows))
    figsize = (num_cols * scale, (num_rows) * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(main_title, fontsize=16) 
    axes = axes.flatten()
    for i, (ax, image) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(image):# tensor
            image = image.cpu().numpy().transpose(1, 2, 0)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)
        else:
            image = np.clip(image, 0, 255)
        ax.imshow(image)
        ax.axis("off")  
        if titles:
            ax.set_title(titles[i], fontsize=8)
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, save_name)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()