import torch.nn as nn
def epoch_adversarial(model, X, y, attack, *args):
    device = next(model.parameters()).device
    X, y = X.to(device), y.to(device)
    delta = attack(model, X, y, *args)
    yp = model(X+delta)
    loss = nn.CrossEntropyLoss()(yp,y)
    total_err = (yp.max(dim=1)[1] != y).sum().item()
    total_loss = loss.item() * X.shape[0]
    return total_err / X.shape[0], total_loss / y.shape[0]