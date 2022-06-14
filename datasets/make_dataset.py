
from torchvision import transforms
from torch.utils.data import Dataset
from .boxdataset import BoxDataset
from .tumordataset import TumorDataset, multitumor2Data, WaveletData

def make_dataset(traindf, input_shape , n_slices, is_train):
    dataloader =  TumorDataset(traindf, input_shape,
                 n_slices = n_slices, is_train = True, normalize = True)
    return dataloader