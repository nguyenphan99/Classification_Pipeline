
import numpy as np
from tqdm import tqdm
from termcolor import cprint
from sklearn.metrics import  f1_score,roc_auc_score
import torch
import torchvision

from .helper_funct import Lossplot, savelog, comp_auc, AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() 
                                  else "cpu")


        
def train_model(model, criterion , optimizer, lr_scheduler, train_dataloader, val_dataloader, config):
    
    """returns trained model"""
    # initialize tracker for minimum validation loss[]
    valid_loss_min = np.Inf
    valid_f1_max = 0.
    valid_AUC_max = 0
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    #global lr
    trainLoss = [] #for build Loss model fig
    validLoss = []
    count = 0

    for epoch in range(0, config.n_epochs):
        
        losses = AverageMeter()
        loss_val = AverageMeter()
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(tqdm(train_dataloader)):
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                if epoch == 0: 
                    optimizer.param_groups[0]['lr'] = config.lr*(batch_idx+1)/len(train_dataloader)
                # importing data and moving to GPU
                img = sample_batched['vol'].to(device, dtype = torch.double)
                label = sample_batched['label'].to(device)

                #model predict
                label_hat = model(img)
                loss = criterion(label_hat, label)
                losses.update(loss.item(), config.batch_size) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if epoch > 0:
                lr_scheduler.step()


        # validate the model #
        #torch.save(model.state_dict(), config.fn_model + str(epoch) + '.pt')

        label_list = []
        label_pred_list = []
        label_prob_list = []
        model.eval()
        for batch_idx, sample_batched in enumerate(tqdm(val_dataloader)):
            with torch.cuda.amp.autocast(enabled=config.use_amp), torch.no_grad():
                img = sample_batched['vol'].to(device, dtype = torch.double)
                label = sample_batched['label'].to(device)
                #model predict
                label_hat = model(img)

                loss = criterion(label_hat, label)
                loss_val.update(loss.item(), config.batch_size)
                
                #f1-score
                label_prob_list.append(label_hat) #auc score
                label_hat = torch.max(label_hat,1)[1].to(device) 
                label_list.append(label)
                label_pred_list.append(label_hat)

        
        label_list, label_pred_list = torch.cat(label_list, 0), torch.cat(label_pred_list, 0)
        f1_scr = f1_score(label_list.cpu(), label_pred_list.cpu(), average = 'macro')
        prob_list = comp_auc(torch.cat(label_prob_list,0), num_classes =2 )
        AUC = roc_auc_score(label_list.cpu(), prob_list[:,1].cpu()) #AUC for 2 classes
        #AUC = roc_auc_score(label_list.cpu(), prob_list.cpu(), multi_class = 'ovr')

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} ({:.6f}) \t f1-score {:.6f} \t AUC {:.6f}'.format(
            epoch, losses.avg, loss_val.val, loss_val.avg, f1_scr, AUC))
        
        #save log
        savelog(epoch,losses.avg, loss_val.avg,f1_scr,AUC,config.lr, config.fn_log)
        trainLoss.append(losses.avg)
        validLoss.append(loss_val.avg)
        
        ## TODO: save the model if validation loss has decreased
        
        if valid_AUC_max < AUC: 
            torch.save(model.state_dict(), config.fn_model + 'AUC_top.pt')
            valid_AUC_max = AUC
        
        if f1_scr > valid_f1_max: 
            torch.save(model.state_dict(), config.fn_model + 'f1scr_top.pt')
            print('Validation f1scr increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_f1_max,
            f1_scr))
            valid_f1_max = f1_scr
            count = 0
        else:
            if epoch >= 40:
                count +=1
            cprint(f'EarlyStopping counter: {count} out of 30', 'yellow')
            if (count == 30 or epoch == config.n_epochs - 1):
                cprint('Early Stop..', 'red')
                torch.save(model.state_dict(), config.fn_model + 'last.pt')
                #Lossplot(trainLoss,validLoss, config.fn_fig)
                exit(-1)
        
        #if loss_val.avg < valid_loss_min:
        #    valid_loss_min = loss_val.avg
        #    torch.save(model.state_dict(), config.fn_model + 'loss.pt')
    return model