import torch
import torch.nn as nn

def compute_grad(model, X, y):
    """ 计算交叉熵对输入的梯度
    
    Args:
        model: the model
        X: the original images, [batch_size, 3, 224, 224], tensor
        y: the labels of X, [batch_size,], tensor
    
    Returns:
        delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
    """
    X.requires_grad = True
    loss = nn.CrossEntropyLoss()(model(X), y)
    loss.backward()
    grad = X.grad.detach().clone()
    return grad.cpu()

def normalized(input_tensor):
    '''归一化'''
    max_value, min_value = input_tensor.max(), input_tensor.min()
    normalized_input_tensor = (input_tensor - min_value) / (max_value - min_value)
    return normalized_input_tensor


def main():
    '''测试'''
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from models.load_model import load_model
    from tools.show_images import show_images
    from data_preprocessor.load_images import load_images
    from tools.get_classes import get_classes_with_index

    images, labels = load_images('./select_images.pth')
    classes = get_classes_with_index(labels)
    model = load_model('resnet50')
    grad = compute_grad(model, images, labels)
    print(grad.shape)
    show_images(normalized(grad), titles=classes, output_path='./data/compute_grad', save_name='grad.png')

if __name__ == '__main__':
    main()
  