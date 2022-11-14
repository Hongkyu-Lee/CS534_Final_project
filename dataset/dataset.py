import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split

SIMPLECAPTCHALEN = 5
ALPHANUMERIC = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 
                'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')

class _SimpleCAPTDataset(Dataset):

    def __init__(self, datalist, path, transform=None):
        '''
            path : path to the dataset
            data_type : train/test/valid
        '''
        self.path = path
        self.transform = transform
        
        # Partition dataset randomly
        # siz
        np.random.seed(0)
        
        self.imgList = datalist 

    def __len__(self):
        # return length
        return len(self.imgList)

    def __getitem__(self, idx):

        _img = Image.open(os.path.join(self.path, self.imgList[idx]))
        _img = _img.convert('RGB')
        _label = torch.zeros(len(ALPHANUMERIC)*SIMPLECAPTCHALEN)
        if (self.transform is not None):
            _img = self.transform(_img)
        for c, i in zip(self.imgList[idx], range(SIMPLECAPTCHALEN)):
            _label[ALPHANUMERIC.index(c) + i*len(ALPHANUMERIC)] = 1
        
        return _img, _label

class SimpleCAPTCHA():
    
    def __init__(self, split, seed, path, transform=None):
        np.random.seed(seed)
        _file_list = (os.listdir(path))
        _file_list.remove(".DS_Store")
        _len = len(_file_list)
        _rand_file_list = np.random.choice(_file_list, size=_len, replace=False)
        _train  = _rand_file_list[0:int(_len*split[0])]
        _valid  = _rand_file_list[int(_len*split[0]):int(_len*split[0])+int(_len*split[1])]
        _test   = _rand_file_list[int(_len*split[0])+int(_len*split[1]):]
        self.train = _SimpleCAPTDataset(_train, path, transform[0])
        self.valid = _SimpleCAPTDataset(_valid, path, transform[1])
        self.test = _SimpleCAPTDataset(_test, path, transform[2])
