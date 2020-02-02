import base64

import cv2
import numpy as np


def get_lr(optimizer, n):
    return optimizer.param_groups[n]['lr'] 

def get_momentum(optimizer, mom_name):
    for param_group in optimizer.param_groups:
        if mom_name == 'momentum':
            return param_group[mom_name]
        elif mom_name == 'betas':
            return param_group[mom_name][0]

def np_to_base64_utf8_str(arr):
    np_buff = arr.tobytes()
    np_buff_b64_bytes = base64.b64encode(np_buff)
    np_buff_base64_utf8_string = np_buff_b64_bytes.decode('utf-8')
    return np_buff_base64_utf8_string

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def set_lr(optimizer, lrs):
    for param_group, lr in zip(optimizer.param_groups, lrs):
        param_group['lr'] = lr
        
def set_momentum(optimizer, mom_name, momentum):
    for param_group in optimizer.param_groups:
        if mom_name == 'momentum':
            param_group[mom_name] = momentum
        elif mom_name == 'betas': 
            param_group[mom_name] = (momentum, 0.999)
        