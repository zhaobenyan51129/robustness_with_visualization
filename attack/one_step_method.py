import os
import sys
import numpy as np
import torch
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.load_model import load_model
from tools.show_images import show_images, plot_distribution, plot_line_chart
from data_preprocessor.load_images import load_images
from tools.get_classes import get_classes_with_index
from algorithms.one_step_attacker import AdversarialAttacksOneStep
import argparse
import matplotlib.pyplot as plt

def parse_args():
    '''参数解析'''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k', type=int, default=10, help='the number of topk')
    return argparser.parse_args()

def save_picturs(images, model, original_classes, delta, grad, norm_grad, output_path):
    '''保存图片
    Args:
        images: 原始图片
        original_classes: 原始图片的标签
        delta: 扰动
        grad: 梯度
        norm_grad: 归一化的梯度
        output_path: 输出路径
        eta: 扰动的阈值
    '''
    attacked_images = torch.clamp(images + delta, 0, 1)
    predict_labels = model(attacked_images)
    pred_classes = get_classes_with_index(predict_labels.argmax(dim=1).cpu().numpy())
    num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(pred_classes, original_classes))
    show_images(norm_grad, titles=original_classes, output_path=output_path, save_name=f'grad.png')
    show_images(delta*100, titles=original_classes, output_path=output_path, save_name=f'delta.png')
    show_images(attacked_images, titles=[label + '/' + pred for label, pred in zip(original_classes, pred_classes)], output_path=output_path, save_name=f'result.png', main_title='{} images are misclassified'.format(num_differences))
    plot_distribution(delta, titles=original_classes, output_path=output_path, save_name=f'delta_distrubution.png')
    plot_distribution(grad, titles=original_classes, output_path=output_path, save_name=f'grad_distrubution.png')
    return num_differences

def perform_attack(images, model, attacker, attacker_func, output_path, original_classes):
    '''执行攻击
    Args:
        attacker: 攻击器
        attacker_func: 攻击函数
        output_folder_name: 输出文件夹的名称
        original_classes: 原始图片的标签
    '''
    delta = attacker_func()
    grad = attacker.grad
    norm_grad = attacker.normalized(grad)
    num_differences = save_picturs(images, model, original_classes, delta, grad, norm_grad, output_path)
    return num_differences

def Attacker(images, labels, model, algorithm, output_path, eta, k):
    '''单步法对抗攻击
    Args:
        images: 原始图片
        labels: 原始图片的标签
        model: 模型
        algorithm: 攻击算法名称
        eta: 扰动的阈值
    '''
    attacker = AdversarialAttacksOneStep(model, images, labels, eta)
    original_classes = get_classes_with_index(labels)

    if algorithm == 'fgsm':
        grad_func = attacker.fgsm
    elif algorithm == 'fgsm_grad_seg_positive':
        attacker.grad = attacker.grad_seg_positive()
        grad_func = lambda: attacker.fgsm(grad=attacker.grad)
    elif algorithm == 'fgsm_grad_seg_negative':
        attacker.grad = attacker.grad_seg_negative()
        grad_func = lambda: attacker.fgsm(grad=attacker.grad)
    elif algorithm == f'fgsm_grad_topk{k}':
        attacker.grad = attacker.grad_topk(k=k)
        grad_func = lambda: attacker.fgsm(grad=attacker.grad)

    elif algorithm == 'fgm':
        grad_func = attacker.fgm
    elif algorithm == 'fgm_grad_seg_positive':
        attacker.grad = attacker.grad_seg_positive()
        grad_func = lambda: attacker.fgm(grad=attacker.grad)
    elif algorithm == 'fgm_grad_seg_negative':
        attacker.grad = attacker.grad_seg_negative()
        grad_func = lambda: attacker.fgm(grad=attacker.grad)
    elif algorithm == f'fgm_grad_topk{k}':
        attacker.grad = attacker.grad_topk(k=k)
        grad_func = lambda: attacker.fgm(grad=attacker.grad)
    
    elif algorithm == 'gaussian_noise':
        attacker.grad = attacker.get_grad()
        grad_func = attacker.gaussian_noise
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')

    num_differences = perform_attack(images, model, attacker, grad_func, output_path, original_classes)
    print(f'{algorithm} has been done, eta{eta}, {num_differences} images are misclassified')
    
    return num_differences

def main(etas, model_str, output_path):
    args = parse_args()
    k = args.k
    images, labels = load_images('./data/images_100.pth')
    images, labels = images[:16].cuda(), labels[:16].cuda()
    # images, labels = load_images('./select_images.pth')
    
    model = load_model(model_str)
    # algorithm_list = ['fgsm', 'fgsm_grad_seg_positive', 'fgsm_grad_seg_negative', f'fgsm_grad_topk{k}', 'fgm', 'fgm_grad_seg_positive', 'fgm_grad_seg_negative', f'fgm_grad_topk{k}', 'gaussian_noise']
    algorithm_list = ['gaussian_noise', 'fgsm', 'fgsm_grad_seg_positive', 'fgsm_grad_seg_negative', f'fgsm_grad_topk{k}']
    save_dict = {}
    for algorithm in algorithm_list:
        save_dict[algorithm] = {}
        num_differences_list = []
        for eta in etas:
            output_path_single = os.path.join(output_path, algorithm, str(eta))
            num_differences = Attacker(images, labels, model, algorithm, output_path_single, eta, k)
            save_dict[algorithm][eta] = num_differences
            num_differences_list.append(num_differences)
        plot_line_chart(etas, num_differences_list, output_path = output_path, save_name = f'{algorithm}.png', title = 'num_images_succeed')
    torch.save(save_dict, os.path.join(output_path, f'save_dict.pth'))

def merge_result(etas, output_path):
    '''合并结果
    Args:
        etas: 扰动阈值的列表
        output_path: 输出路径'''
    save_dict = torch.load(os.path.join(output_path, 'save_dict.pth'))
    plt.figure()

    for algorithm, data in save_dict.items():
        num_differences_list = [data[eta] for eta in etas]
        plt.plot(etas, num_differences_list, label=algorithm)

    plt.legend()

    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, 'merged.png')
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    etas = np.arange(0, 0.2, 0.01)
    model_str = 'vit_b_16'  # 'resnet50'
    # model_str = 'resnet50'
    output_path = f'./data/one_step_attack_{model_str}'
    main(etas, model_str, output_path)
    merge_result(etas, output_path)


    # 调用方式 python attack/one_step_method.py --k 2000 