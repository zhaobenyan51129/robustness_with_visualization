import torch
import torch.nn as nn

def fgm(model, X, y, eta):
    """ Construct FGSM adversarial examples on the examples X
    
    Args:
        model: the model
        X: the original images, [batch_size, 3, 224, 224], tensor
        y: the labels of X, [batch_size,], tensor
        eta: the eta (input variation parameter)   
    Returns:
        delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
    """
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    g = delta.grad.detach().clone()
    norm_g = torch.norm(g, p=2)
    return eta * (g / norm_g)


def main():
    '''测试'''
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from models.load_model import load_model
    from tools.show_images import show_images
    from datasets.load_images import load_images
    from tools.get_classes import get_classes_with_index

    images, labels = load_images('./select_images.pth')
    # image = images[0]
    # label = labels[0]
    classes = get_classes_with_index(labels)
    model = load_model('resnet50')
    eta = 0.01
    delta = fgm(model, images, labels, eta)
    print(delta.shape)
    show_images(images, titles=classes, output_path='./data/fgm', save_name='fgm.png')

if __name__ == '__main__':
    main()
  