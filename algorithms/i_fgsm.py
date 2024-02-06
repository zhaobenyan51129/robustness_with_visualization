import torch
import torch.nn as nn

def i_fgsm(model, X, y, alpha, eta, num_steps):
    """ Construct I-FGSM adversarial examples on the examples X
    
    Args:
        model: the model
        X: the original images, [batch_size, 3, 224, 224], tensor
        y: the labels of X, [batch_size,], tensor
        alpha: 扰动的步长
        eta: 扰动阈值  
        num_steps: 迭代次数  
    Returns:
        delta: the adversarial perturbation, [batch_size, 3, 224, 224], tensor     
    """
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_steps):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        grad = delta.grad.detach().clone()
        delta.data = delta + alpha * grad.sign()
        delta.data = torch.clamp(delta, -eta, eta)
        delta.grad.zero_()
    return delta


def main():
    '''测试'''
    import sys
    sys.path.append('C:\\Users\\19086\\Desktop\\experince\\robustness_with_visualization')
    from models.load_model import load_model
    from tools.show_images import show_images
    from datasets.load_images import load_images
    from tools.get_classes import get_classes_with_index

    images, labels = load_images('./select_images.pth')
    original_classes = get_classes_with_index(labels)
    model = load_model('resnet50')
    eta = 0.01
    alpha = 0.01
    num_steps = 10
    delta = i_fgsm(model, images, labels, alpha, eta, num_steps).detach()
    attcked_images = torch.clamp(images + delta, 0, 1)
    predict_labels = model(attcked_images)
    predict_classes = get_classes_with_index(predict_labels.argmax(dim=1).cpu().numpy())
    num_differences = sum(pred_class != orig_class for pred_class, orig_class in zip(predict_labels.argmax(dim=1).cpu().numpy(), labels))
    titles = [label + '/' + pred for label, pred in zip(original_classes, predict_classes)]
    show_images(attcked_images, titles=titles, output_path='./data/i_fgsm', save_name='i_fgsm.png', main_title='{} images are misclassified'.format(num_differences))

if __name__ == '__main__':
    main()
  