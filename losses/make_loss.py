from .loss import FocalLoss, LabelSmoothing, CrossEntropyLabelSmooth
from torch.nn import CrossEntropyLoss
from .PolyLoss import poly1_cross_entropy

def make_loss(config):
    if config.loss_type == 'crossentropy':
        loss_func = CrossEntropyLoss()
    elif config.loss_type == 'polyloss':
        loss_func = poly1_cross_entropy()
    elif config.loss_type == 'focal_loss':
        loss_func = FocalLoss()
    elif config.loss_type == 'combined_loss':
        loss_func = CrossEntropyLoss() + poly1_cross_entropy() + FocalLoss()
    elif config.loss_type == 'cross_entropy_with_logit':
        loss_func = nn.BCEWithLogitsLoss()
    elif config.loss_type == 'label_smoothing':
        loss_func = LabelSmoothing()
    elif config.loss_type =='cross_entropy_label_smoothing':
        loss_func = CrossEntropyLabelSmooth()
    return loss_func