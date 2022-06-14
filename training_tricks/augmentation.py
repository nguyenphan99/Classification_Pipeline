'''
Modified from st2-VinBrain
'''
import numpy as np
import torch
import sys
sys.path.append('/vinbrain/st2/breast_cancer/classification/heatmap/bi_rads_classification/FMix')
from fmix import sample_mask, make_low_freq_image, binarise_mask
from fmix import sample_mask
import random
def mixup_data(x, y, gpu, mixup_prob=0.5, alpha=1.0):
    
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if random.uniform(0, 1) < mixup_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(gpu)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     if lam < 0.5:
#         return criterion(pred, torch.max(y_a,y_b))
#     else:
#         return criterion(pred, torch.max(y_a,y_b))
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
### ted model ###

def rand_bbox_original(size, lam):
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

def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.7)
    data = lam*data + (1-lam)*shuffled_data
    targets = (target, shuffled_target, lam)

    return data, targets


def fmix_original(data, targets, alpha, decay_power, shape, max_soft=0.0, device='cuda'):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft)
    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    mask = torch.from_numpy(mask)
    mask = mask.to(device)
    x1 = mask*data
    x2 = (1-mask)*shuffled_data
    targets=(targets, shuffled_targets, lam)
    
    return (x1+x2), targets

def two_view_mixup_data(x1, x2, y, gpu, mixup_prob=0.5, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if random.uniform(0, 1) < mixup_prob:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(gpu)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, y_a, y_b, lam


### aug two view ###
def cutmix(image_cc, image_mlo, target, alpha):
    
    indices = torch.randperm(image_cc.size(0))
    shuffled_data = image_cc[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_cc.size(), lam)
    
    new_data_cc = image_cc.clone()
    new_data_cc[:, :, bby1:bby2, bbx1:bbx2] = image_cc[indices, :, bby1:bby2, bbx1:bbx2]
    
    new_data_mlo = image_mlo.clone()
    new_data_mlo[:, :, bby1:bby2, bbx1:bbx2] = image_mlo[indices, :, bby1:bby2, bbx1:bbx2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_cc.size()[-1] * image_cc.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data_cc, new_data_mlo, targets


def cutmix_original(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox_original(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


def fmix(image_cc, image_mlo, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False, device='cpu'):
    
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(image_cc.size(0))
    
    shuffled_data_cc = image_cc[indices]
    shuffled_data_mlo = image_mlo[indices]
    
    shuffled_targets = targets[indices]
    
    x1_cc = torch.from_numpy(mask).to(device)*shuffled_data_cc
    x1_mlo = torch.from_numpy(mask).to(device)*shuffled_data_mlo
    
    x2_cc = torch.from_numpy(1-mask).to(device)*shuffled_data_cc
    x2_mlo = torch.from_numpy(1-mask).to(device)*shuffled_data_mlo
    
    targets= (targets, shuffled_targets, lam)
    
    return (x1_cc+x2_cc), (x2_mlo+x2_mlo), targets

def rand_bbox(size, lam):
    
    W = size[0]
    H = size[1]
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

