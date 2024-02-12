import torch
import torch.nn as nn

def pgd(model, X, y, alpha, eta, num_steps):
    """ Construct FGSM adversarial examples on the examples X
    Args:
        model: the model
        X: the original images, [batch_size, 3, 224, 224], tensor
        y: the labels of X, [batch_size,], tensor
        eta: the epsilon (input variation parameter)
        alpha: step size
        num_steps: number of iterations
    """
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_steps):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-eta,eta)
        delta.grad.zero_()
    return delta.detach()

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
    epsilon = 0.1
    alpha = 1e4
    num_iter = 20
    delta = pgd(model, images, labels, epsilon, alpha, num_iter)
    print(delta)
    show_images(images, titles=classes, output_path='./data/pgd', save_name='pgd.png')

if __name__ == '__main__':
    main()