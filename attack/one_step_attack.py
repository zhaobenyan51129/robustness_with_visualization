import sys
sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
from models.load_model import load_model
from tools.show_images import show_images
from datasets.load_images import load_images
from tools.get_classes import get_classes_with_index
from algorithms.one_step_attacker import AdversarialAttacksOneStep

def save_picturs(images, original_classes, delta, norm_grad, output_path, eta):
    attacked_images = images + delta
    predict_labels = model(attacked_images)
    pred_classes = get_classes_with_index(predict_labels.argmax(dim=1).cpu().numpy())
    num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(pred_classes, original_classes))
    show_images(norm_grad, titles=original_classes, output_path=output_path, save_name=f'grad_eta{eta}.png')
    show_images(delta*100, titles=original_classes, output_path=output_path, save_name=f'delta_eta{eta}.png')
    show_images(attacked_images, titles=[label + '/' + pred for label, pred in zip(original_classes, pred_classes)], output_path=output_path, save_name=f'result_eta{eta}.png', main_title='{} images are misclassified'.format(num_differences))


def Attacker(images, labels, model, algorithm_list, eta=0.01):
    '''单步法对抗攻击
    Args:
        images: 原始图片
        labels: 原始图片的标签
        model: 模型
        algorithm_list: 攻击算法名称
        eta: eta (input variation parameter)
    '''
    attacker = AdversarialAttacksOneStep(model, images, labels, eta)
    original_classes = get_classes_with_index(labels)

    if 'fgsm' in algorithm_list:
        delta = attacker.fgsm()
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = './data/onestep_attack/fgsm'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print('fgsm has been done')
    if 'fgm' in algorithm_list:
        delta = attacker.fgm()
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = './data/onestep_attack/fgm'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print('fgm has been done')
    if 'gaussian_noise' in algorithm_list:
        delta = attacker.gaussian_noise()
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = './data/onestep_attack/gaussian_noise'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print('gaussian_noise has been done')
    if 'fgsm_grad_seg_positive' in algorithm_list:
        attacker.grad = attacker.grad_seg_positive()
        delta = attacker.fgsm(grad=attacker.grad)
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = './data/onestep_attack/grad_seg_positive'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print('fgsm_grad_seg_positive has been done')
    if 'fgsm_grad_seg_negative' in algorithm_list:
        attacker.grad = attacker.grad_seg_negative()
        delta = attacker.fgsm(grad=attacker.grad)
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = './data/onestep_attack/grad_seg_negative'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print('fgsm_grad_seg_negative has been done')
    if 'fgsm_grad_topk' in algorithm_list:
        k = 10000
        attacker.grad = attacker.grad_topk(k=k)
        delta = attacker.fgsm(grad=attacker.grad)
        grad = attacker.grad
        norm_grad = attacker.normalized(grad)
        output_path = f'./data/onestep_attack/grad_top{k}'
        save_picturs(images, original_classes, delta, norm_grad, output_path, eta)
        print(f'fgsm_grad_top{k} has been done')
    print('attacked images has been saved')

if __name__ == '__main__':
    images, labels = load_images('./select_images.pth')
    model = load_model('resnet50' )
    algorithm_list = ['fgsm', 'fgm', 'gaussian_noise', 'fgsm_grad_seg_positive', 'fgsm_grad_seg_negative', 'fgsm_grad_topk']

    Attacker(images, labels, model, algorithm_list, eta=0.01)