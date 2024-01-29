
def special_ruled_lr(epoch):
    lr = 0.9**((epoch)//100)    
    return lr

def big_start_special_ruled_lr(epoch):
    if epoch <= 50:
        lr = 100
    else:
        lr = 0.9**((epoch)//100)    
    return lr

def exp_lr(epoch):
    lr = 0.1 * 0.99**epoch
    if lr <= 1e-4:
        lr = 1e-4
    return lr

def big_start_exp_lr(epoch):
    if epoch < 5:
        lr = 1e3
    else:
        lr = 0.1 * 0.99**epoch
    if lr <= 1e-4:
        lr = 1e-4
    return lr