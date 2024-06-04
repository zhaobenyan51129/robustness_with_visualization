import sys
import torch
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from models.load_model import load_model
from tools.show_images import show_images, plot_distrubution
from data_preprocessor.load_images import load_images
from tools.get_classes import get_classes_with_index
from algorithms.multi_step_attacker import AdversarialAttacksMultiStep
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
    output_path = f'./data/multistep_attack/{output_folder_name}'
    num_differences = save_picturs(images, model, original_classes, delta, grad, norm_grad, output_path, eta)
    return num_differences

def Attacker(images, labels, model, algorithm_list, alpha, eta, k, num_steps):
    '''多步法对抗攻击
    Args:
        images: 原始图片
        labels: 原始图片的标签
        model: 模型
        algorithm_list: 攻击算法名称
        alpha: 步长
        eta: 扰动的阈值
        k: topk选择的张数
        num_steps: 迭代次数
    '''
    attacker = AdversarialAttacksMultiStep(model, images, labels, alpha, eta, num_steps)
    original_classes = get_classes_with_index(labels)

    for algorithm in algorithm_list:
        if algorithm == 'i_fgsm' or algorithm == 'i_fgsm_positive' or algorithm == 'i_fgsm_negative' or algorithm == f'i_fgsm_topk{k}':
            grad_func = lambda: attacker.i_fgsm(grad_mode=algorithm, k=k)

        elif algorithm == 'i_fgm' or algorithm == 'i_fgm_positive' or algorithm == 'i_fgm_negative' or algorithm == f'i_fgm_topk{k}':
            grad_func = lambda: attacker.i_fgm(grad_mode=algorithm, k=k)
        
        elif algorithm == 'pgd' or algorithm == 'pgd_positive' or algorithm == 'pgd_negative' or algorithm == f'pgd_topk{k}':
            grad_func = lambda: attacker.pgd(grad_mode=algorithm, k=k)
        
        elif algorithm == 'i_gaussian_noise':
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
    model = load_model('resnet50' )
    model_name = type(model).__name__
    print("模型名称:", model_name)  
    # model = load_model('vit_b_16')
    alpha = 0.01
    num_steps = 10
    algorithm_list = ['i_fgsm', 'i_fgsm_positive', 'i_fgsm_negative', f'i_fgsm_topk{k}', 'i_fgm', 'i_fgm_positive', 'i_fgm_negative', f'i_fgm_topk{k}', 'pgd', 'pgd_positive', 'pgd_negative', f'pgd_topk{k}', 'i_gaussian_noise']
    Attacker(images, labels, model, algorithm_list, alpha, eta, k, num_steps)

if __name__ == '__main__':
    main()
    
    