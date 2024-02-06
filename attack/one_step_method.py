import sys

import torch
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from models.load_model import load_model
from tools.show_images import show_images, plot_distrubution
from datasets.load_images import load_images
from tools.get_classes import get_classes_with_index
from algorithms.one_step_attacker import AdversarialAttacksOneStep
import argparse

def parse_args():
    '''参数解析'''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k', type=int, default=10, help='the number of topk')
    argparser.add_argument('--eta', type=float, default=0.01, help='Disturbance threshold')
    return argparser.parse_args()

def save_picturs(images, model, original_classes, delta, grad, norm_grad, output_path, eta):
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
    show_images(norm_grad, titles=original_classes, output_path=output_path, save_name=f'grad_eta{eta}.png')
    show_images(delta*100, titles=original_classes, output_path=output_path, save_name=f'delta_eta{eta}.png')
    show_images(attacked_images, titles=[label + '/' + pred for label, pred in zip(original_classes, pred_classes)], output_path=output_path, save_name=f'result_eta{eta}.png', main_title='{} images are misclassified'.format(num_differences))
    plot_distrubution(delta, titles=original_classes, output_path=output_path, save_name=f'delta_distrubution_eta{eta}.png')
    plot_distrubution(grad, titles=original_classes, output_path=output_path, save_name=f'grad_distrubution_eta{eta}.png')
    return num_differences

def perform_attack(images, model, attacker, attacker_func, output_folder_name, original_classes, eta):
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
    output_path = f'./data/onestep_attack/{output_folder_name}'
    num_differences = save_picturs(images, model, original_classes, delta, grad, norm_grad, output_path, eta)
    return num_differences

def Attacker(images, labels, model, algorithm_list, eta, k):
    '''单步法对抗攻击
    Args:
        images: 原始图片
        labels: 原始图片的标签
        model: 模型
        algorithm_list: 攻击算法名称
        eta: 扰动的阈值
    '''
    attacker = AdversarialAttacksOneStep(model, images, labels, eta)
    original_classes = get_classes_with_index(labels)

    for algorithm in algorithm_list:
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
            grad_func = attacker.gaussian_noise
        else:
            raise ValueError(f'Unknown algorithm: {algorithm}')

        num_differences = perform_attack(images, model, attacker, grad_func, algorithm, original_classes, eta)
        print(f'{algorithm} has been done, {num_differences} images are misclassified')


def main():
    args = parse_args()
    k = args.k
    eta = args.eta
    images, labels = load_images('./select_images.pth')
    # model = load_model('resnet50' )
    model = load_model('vit_b_16')
    algorithm_list = ['fgsm', 'fgsm_grad_seg_positive', 'fgsm_grad_seg_negative', f'fgsm_grad_topk{k}', 'fgm', 'fgm_grad_seg_positive', 'fgm_grad_seg_negative', f'fgm_grad_topk{k}', 'gaussian_noise']
    Attacker(images, labels, model, algorithm_list, eta, k)

if __name__ == '__main__':
    main()
    # 调用方式 python attack/one_step_method.py --k 10 --eta 0.01