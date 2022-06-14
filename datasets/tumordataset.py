"""
Create on 16/03/2022
@author: Tran Bao Sam
This dataset uses for HCC/ABN cls base on 2D model
"""
#__Import Libraries__
import numpy as np
import random
import math 


import torch
from torchvision import transforms
from torch.utils.data import Dataset
import monai
from scipy.interpolate import interp1d
import pywt


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transform(input_shape, is_train):
    if is_train:
        return monai.transforms.Compose([monai.transforms.RandRotate(range_x=30, range_y=30),
                                        monai.transforms.Flip(spatial_axis=0),
                                        monai.transforms.Resize(input_shape, size_mode='all'),
                                        monai.transforms.ToTensor()]) 
    else:
        return monai.transforms.Compose([monai.transforms.ToTensor()]) #monai.transforms.Resize(input_shape, size_mode='all'),
        
    
def get_label():
    return {
        "abn" : 0,
        "hcc" : 1,
    }

def HUnormalize(vol):
    """Normalize the volume"""
    min = 0
    max = 400
    vol = (vol - min) / (max - min)
    vol = vol.astype("float32")
    return vol

def crt_path(pid,name):
    #return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/tumor_crop/" + name + "/Venous_Phase.npy" 
    #return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/update_test_set/bounding_box_data/intersect/" + pid + "/Arterial_" + name +".npy" #for bouding box data
    #return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/update_test_set/tumor_crop/intersect/" + pid + "/Venous_" + name +".npy" #for tumor crop
    return "/vinbrain/samtb/Liver_Tumor_Cls/dataset/update_test_set/tumor_crop_oversize/intersect/" + pid + "/Venous_" + name + ".npy" 

def crt_path_2phase(pid, name): #name
    #folder = "/vinbrain/samtb/Liver_Tumor_Cls/dataset/tumor_crop_224_224_128/" #data cũ 
    #folder = "/vinbrain/samtb/Liver_Tumor_Cls/dataset/new_mask/tumor_crop/" #data mới
    #folder = "/vinbrain/samtb/Liver_Tumor_Cls/dataset/update_test_set/tumor_crop/union/" #test dataset
    folder = "/vinbrain/samtb/Liver_Tumor_Cls/dataset/update_test_set/bounding_box_data/union/"
    #return folder + pid + "/Venous_Phase.npy", folder + pid + "/Arterial_Phase.npy"
    return folder + pid + "/Venous_" + name + ".npy", folder + pid + "/Arterial_" + name + ".npy"

"""
Input: 1 3D imgs
Output: 1 2D imgs (124,124,15) [15 channels]

Approach 1: Take imgs in tumor crop
"""
class TumorDataset(Dataset):
    def __init__(self,dataframe, input_shape = (224,224),
                 n_slices = 64,
                 is_train = True, normalize = True):
        self.dataframe = dataframe
        self.input_shape = input_shape
        self.n_slices = n_slices       
        self.is_train = is_train        
        self.normalize = normalize
        self.get_label = get_label()

        self.tfms = get_transform(self.input_shape, self.is_train)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        #return image, label
        info = self.dataframe.iloc[index]
        #label = info["label"]
        #volpath = crt_path(info["name"])
        
        label = self.get_label[info["global_type"]]
        if info["date"]=='NG':
            volpath = crt_path(info["patient_id"], info["name"])
        else:
            volpath = crt_path(info["patient_id"], info["date"] +"_"+ info["name"])
        
         
        """3D IMG LOAD"""
        vol = np.load(volpath,"r") #the size is (H x W x n_slices)
        vol = vol.T #to obtain the size of (n_slices x H x W) (64,124,124)
        
        """
        #wavelet transform
        coeffs2 = pywt.dwt2(vol, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        #concat 4 transform imgs to 1 img (shape: 224, 224, 128)
        LL, LH, HL, HH = np.swapaxes(LL,2,0), np.swapaxes(LH,2,0), np.swapaxes(HL,2,0), np.swapaxes(HH,2,0)
        wl_line1 = np.concatenate((LL[:-2,:-2,:],LH[2:,2:,:]), axis = 0)
        wl_line2 = np.concatenate((HL[:-2,:-2,:],HH[2:,2:,:]), axis = 0)
        wl_img = np.concatenate((wl_line1, wl_line2), axis = 1)
        wl_img = np.swapaxes(wl_img,2,0)
        """
        
        #get random slices
        if self.normalize:
            vol = HUnormalize(vol)
            #wl_img = HUnormalize(wl_img)
        
        vol = self.tfms(vol)
        #wl_img = self.tfms(wl_img)
        
        sample = {"vol": vol, "label": label} #, "wl_vol": wl_img
        return sample
    
class multitumor2Data(Dataset):
    def __init__(self,dataframe, 
                 input_shape = (124,124),
                 n_slices = 64,
                 is_train = True,
                 normalize = True, 
                 fusion = "add"):
        self.dataframe = dataframe
        self.input_shape = input_shape
        self.n_slices = n_slices
        self.is_train = is_train
        self.tfms = get_transform(self.input_shape, self.is_train)
        self.normalize = normalize
        self.fusion = fusion
        self.get_label = get_label()
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        #return image, label
        info = self.dataframe.iloc[index]
        venpath, artpath= crt_path_2phase(info["patient_id"], info["name"]) #info["patient_id"], info["name"]
        #venpath, artpath = info["ven_path"], info["art_path"]
        #label = info["label"]
        label = self.get_label[info["global_type"]]
        
        """3D IMG TRANS"""
        venvol = np.load(venpath,"r").T #the size is (H x W x n_slices) -> (n_slices x H x W)
        artvol = np.load(artpath, "r").T
        
        #for chexnet model
        #m_ven = interp1d([venvol.min(),venvol.max()],[0,255]) #map from value range from another range
        #m_art = interp1d([artvol.min(),artvol.max()],[0,255])       
        #venvol = np.round(m_ven(venvol))
        #artvol = np.round(m_art(artvol))
        
        #if math.isnan(venvol.max()):
        #    print(venpath)
        #if math.isnan(artvol.max()):
        #    print(artpath)
        
        #clip data
        """
        venvol = np.clip(venvol,-20,400)
        artvol = np.clip(artvol,-20,400)
        venvol = np.where((venvol<20) &(venvol!=0) ,20,venvol)
        artvol = np.where((artvol<20) &(artvol!=0) ,20,artvol)
        """
        #get mean features of venvol and artvol        
        if self.fusion == "add":
            vol = np.add(venvol,artvol)
        elif self.fusion == "mean":
            vol = np.add(venvol,artvol)/2
        
        if self.normalize == True:
            vol = HUnormalize(vol)
            
        vol = self.tfms(vol)
        
        sample = {"vol": vol, "label": label} 
        return sample
    
class WaveletData(Dataset):
    def __init__(self,dataframe, 
                 input_shape = (124,124),
                 n_slices = 64,
                 is_train = True,
                 normalize = True, 
                 fusion = "add"):
        self.dataframe = dataframe
        self.input_shape = input_shape
        self.n_slices = n_slices
        self.is_train = is_train
        self.tfms = get_transform(self.input_shape, self.is_train)
        self.tfms_wl = get_transform(self.input_shape, self.is_train)
        self.normalize = normalize
        self.fusion = fusion
        self.get_label = get_label()
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        #return image, label
        info = self.dataframe.iloc[index]
        venpath, artpath= crt_path_2phase( info["name"]) #info["patient_id"], info["name"]
        #venpath, artpath = info["ven_path"], info["art_path"]
        label = info["label"]
        #label = self.get_label[info["global_type"]]
        
        """3D IMG TRANS"""
        venvol = np.load(venpath,"r").T #the size is (H x W x n_slices) -> (n_slices x H x W)
        artvol = np.load(artpath, "r").T
        
        
        #get mean features of venvol and artvol        
        if self.fusion == "add":
            vol = np.add(venvol,artvol)
        elif self.fusion == "mean":
            vol = np.add(venvol,artvol)/2
        
        #wavelet transform
        
        coeffs2 = pywt.dwt2(vol, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        #concat 4 transform imgs to 1 img (shape: 224, 224, 128)
        LL, LH, HL, HH = np.swapaxes(LL,2,0), np.swapaxes(LH,2,0), np.swapaxes(HL,2,0), np.swapaxes(HH,2,0)
        wl_line1 = np.concatenate((LL[:-2,:-2,:],LH[2:,2:,:]), axis = 0)
        wl_line2 = np.concatenate((HL[:-2,:-2,:],HH[2:,2:,:]), axis = 0)
        wl_img = np.concatenate((wl_line1, wl_line2), axis = 1)
        wl_img = np.swapaxes(wl_img,2,0)
        
        #concat 4 transform imgs to 1 img (shape: 128*4 , 114, 114)
        #wl_img = np.concatenate((LL,LH,HL,HH), axis = 0)
        
        if self.normalize == True:
            vol = HUnormalize(vol)
            wl_img = HUnormalize(wl_img)
        
        #print("Before transform: ", wl_img.shape)
        vol = self.tfms(vol)
        wl_img = self.tfms_wl(wl_img)
        #print("After transform: ", wl_img.shape)
        sample = {"vol": vol, "wl_vol" : wl_img , "label": label}  
        #sample = {"vol": wl_img , "label": label} 
        return sample
    
class CombineRadiomic(Dataset):
    def __init__(self,dataframe, pyradiodf, input_shape = (224,224),
                 n_slices = 64,
                 is_train = True, normalize = True):
        self.dataframe = dataframe
        self.pyradiodf = pyradiodf
        self.input_shape = input_shape
        self.n_slices = n_slices       
        self.is_train = is_train        
        self.normalize = normalize
        self.get_label = get_label()

        self.tfms = get_transform(self.input_shape, self.is_train)

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        #return image, label, pyradiomic feature
        info = self.dataframe.iloc[index]     
        label = self.get_label[info["global_type"]]
        volpath = crt_path(info["patient_id"], info["name"])
        #print("ID: ", info["patient_id"] + "_" + info["name"])
        pyradio_fea = self.pyradiodf[self.pyradiodf["ID"] == (info["patient_id"] + "_" + info["name"])].iloc[0,1:].values
        pyradio_fea = pyradio_fea.astype("float64")
        pyradio_fea = torch.from_numpy(pyradio_fea)
         
        """3D IMG LOAD"""
        vol = np.load(volpath,"r") #the size is (H x W x n_slices)
        vol = vol.T #to obtain the size of (n_slices x H x W) (64,124,124)
        
        #get random slices
        if self.normalize:
            vol = HUnormalize(vol)
        
        vol = self.tfms(vol)
        
        sample = {"vol": vol, "label": label, "pyradio_fea": pyradio_fea } 
        return sample