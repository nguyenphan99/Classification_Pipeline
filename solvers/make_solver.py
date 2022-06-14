from torch import nn, optim
import torch
from ranger21 import Ranger21

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, OneCycleLR,ReduceLROnPlateau,OneCycleLR
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

def make_solver(config, model, train_dataloader):
    if config.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer_type == 'Ranger21':
        optimizer = Ranger21(model.parameters(),lr=config.lr,num_epochs=config.n_epochs,use_madgrad=False,use_warmup=True,\
                             use_cheb=False, num_batches_per_epoch=len(train_dataloader), weight_decay=1e-5, warmdown_min_lr=3e-8)
    else:
        optimizer = optim.AdamW(model.parameters(),lr=config.lr)
        
    if config.lr_scheduler_type == 'onecyclelr':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=config.n_epochs)
    elif config.lr_scheduler_type =='ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, verbose=True,factor=0.1)
    elif config.lr_scheduler_type=='Custom_Scheduler':
        lr_scheduler = Custom_Scheduler(optimizer,**CFG.SCHEDULER_PARAMS)
    elif config.lr_scheduler_type =='CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr,last_epoch=-1)
    elif config.lr_scheduler_type =='CosineAnnealingWarmRestarts':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=2, eta_min=CFG.min_lr,last_epoch=-1)
    elif config.lr_scheduler_type == 'MultiStepLR':
        lr_scheduler = MultiStepLR(optimizer, milestones=config.milestones)
    elif config.lr_scheduler_type =='OneCycleLR':
        lr_scheduler = OneCycleLR(optimizer, max_lr=CFG.max_lr, total_steps=CFG.total_steps)    
    
    return optimizer, lr_scheduler
