from .resnet import resnet34
from .densenet import densenet121
from .classifier import classifier

def make_model(config):
    if config.model_type == 'classifier':
        model = classifier(in_channels = config.in_channels, image_size = (config.input_size,config.input_size))
    elif config.model_type == 'densnet':
        model = densnet121()
    elif config.model_type == 'resnet':
        model = resnet34()
    
    return model