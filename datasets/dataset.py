"""
Create on 08/03/2022
@author: Tran Bao Sam
This dataset uses for HCC/ABN/NFD cls base on 2D model
"""
#__Import Libraries__
import numpy as np
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import monai

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transform(input_shape):
        return monai.transforms.Compose([monai.transforms.Resize(input_shape, size_mode='all'),
                                    monai.transforms.ToTensor()])

def HUnormalize(vol):
    """Normalize the volume"""
    min = 0
    max = 400
    vol = (vol - min) / (max - min)
    vol = vol.astype("float32")
    return vol

def crt_path(name, label):
    if label == 0: #return img
        return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/tumor_crop/" + name + "/Venous_Phase_pad.npy" #for NFD
    else:
        return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/tumor_crop/" + name + "/Venous_Phase.npy" 

"""
Input: 1 3D imgs
Output: 1 2D imgs (224,244,15) [15 channels]

Approach 1: Take imgs in tumor crop
"""
class TumorDataset(Dataset):
    def __init__(self,dataframe, input_shape = (224,224),
                 n_slices = 15,
                 is_train = True, normalize = True):
        self.dataframe = dataframe
        self.input_shape = input_shape
        self.n_slices = n_slices
        
        self.is_train = is_train        
        self.normalize = normalize

        self.tfms = get_transform(self.input_shape)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        #return image, label
        info = self.dataframe.iloc[index]
        label = info["label"]
        volpath = crt_path(info["name"], int(label))
        
         
        """3D IMG LOAD"""
        vol = np.load(volpath,"r") #the size is (H x W x n_slices)
        vol = vol[:,:,20:(vol.shape[2]-20)] #for the box img
        vol = vol.T #to obtain the size of (n_slices x H x W)
        
        #np.random.seed(15)
        #vol = np.clip(vol,50,400)
        #get random slices
        if self.is_train:
            vol = vol[np.random.choice(vol.shape[0], size = self.n_slices, replace=vol.shape[0]<self.n_slices), ...]
        else:
            vol = vol[np.linspace(0, vol.shape[0] - 1, self.n_slices, dtype=int), ...]
        
        if self.normalize:
            vol = HUnormalize(vol)
        
        vol = self.tfms(vol)
        
        sample = {"vol": vol, "label": label} 
        return sample

    
