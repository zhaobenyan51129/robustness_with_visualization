import torch
import torch.nn as nn

def fgsm(model, X, y, eta):
    """ Construct FGSM adversarial examples on the examples X
    
    Args:
        model: the model
        X: the original images, [batch_size, 3, 224, 224], tensor
        y: the labels of X, [batch_size,], tensor
        eta: 扰动阈值 
    Returns:
        delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
    """
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    grad = delta.grad.detach().clone()
    perturbation = eta * grad.sign()
    # 下面是单张单张图片地处理
    # loss_list = []
    # grad_list = []
    # for image, label in zip(X, y):
    #     image = image.unsqueeze(0)
    #     label = label.unsqueeze(0)
    #     delta = torch.zeros_like(image, requires_grad=True)
    #     loss = nn.CrossEntropyLoss()(model(image + delta), label)
    #     loss.backward()
    #     loss_list.append(loss.item())
    #     grad_list.append(delta.grad.detach().clone())
    # print(sum(loss_list)/len(loss_list))
    return perturbation


def main():
    '''测试'''
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from models.load_model import load_model
    from tools.show_images import show_images
    from data_preprocessor.load_images import load_images
    from tools.get_classes import get_classes_with_index

    images, labels = load_images('./select_images.pth')
    # image = images[0]
    # label = labels[0]
    classes = get_classes_with_index(labels)
    model = load_model('resnet50')
    eta = 0.01
    delta = fgsm(model, images, labels, eta)
    # print(delta)
    show_images(images, titles=classes, output_path='./data/fgsm', save_name='fgsm.png')

if __name__ == '__main__':
    main()
  