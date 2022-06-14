import matplotlib.pyplot as plt
import torch


def Lossplot(trainLoss, validLoss, folder ):
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(folder, facecolor = "w")
    plt.show()

def savelog(epoch,losstrain, lossval,f1scr,AUC, lr, name ):
    log = '\nEpoch: '+ str(epoch) + ' \tTraining Loss: '+ str(losstrain) + ' \tValidation Loss: ' + str(lossval) + ' \t f1-score: ' +str(f1scr) + ' \t AUC: ' +str(AUC)  + ' \t learning rate: ' +str(lr) 
    f = open(name, "a")
    f.write(log)
    f.close()


def comp_auc(label_prob_list, num_classes = 2):
    m = torch.nn.Softmax(dim=0)
    prob_list = []
    for i in range(len(label_prob_list)):
        prob_list.append(m(label_prob_list[i]).tolist())
        
    prob_list = torch.Tensor(prob_list)
    row_sums = torch.sum(prob_list, 1) # normalization 
    row_sums = row_sums.reshape(-1,1)
    row_sums = row_sums.repeat(1, num_classes) # expand to same size as number output classes : 3
    prob_list = torch.div( prob_list , row_sums ) # these should be histograms
    return prob_list

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count