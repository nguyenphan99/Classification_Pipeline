import math
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from .densenet import densenet121


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

#model for 2 backbone
class multiImgClassifier(nn.Module):
    def __init__(self, pretrained = True, in_channels = 64, 
                 image_size = (124,124), name_model = "densenet121", 
                 num_classes = 2):
        super(multiImgClassifier, self).__init__()
        
        self.name_model = name_model
        if pretrained is True:
            self.backbone = densenet121(pretrained = True, in_channels = in_channels)
        self.fc = nn.Linear(1000, 512) #num_classes
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,num_classes)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x, y):
        x = self.backbone(x)
        y = self.backbone(y)
        
        x = self.fc(x)
        y = self.fc(y)
        
        res = torch.cat((x,y), axis = 1)
        #res = self.dropout(res)
        res = self.fc1(res)
        res = self.dropout(res)
        res = self.fc2(res)
        return res

#the highest perform model

class classifier(nn.Module):
    def __init__(self, pretrained = True, in_channels = 64, 
                 image_size = (124,124), name_model = "densenet121", 
                 num_classes = 2):
        super(classifier, self).__init__()
        
        self.name_model = name_model
        if pretrained is True:
            self.backbone = densenet121(pretrained = True, in_channels = in_channels)
        self.fc = nn.Linear(1000, 512) #num_classes
        self.fc2 = nn.Linear(512,num_classes)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x