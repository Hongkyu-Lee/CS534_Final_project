import os
import cv2
import numpy as np
import random
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
        _img = _img.convert('L')
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
        if transform is not None:
            self.train = _SimpleCAPTDataset(_train, path, transform[0])
            self.valid = _SimpleCAPTDataset(_valid, path, transform[1])
            self.test = _SimpleCAPTDataset(_test, path, transform[2])
        else:
            self.train = _SimpleCAPTDataset(_train, path)
            self.valid = _SimpleCAPTDataset(_valid, path)
            self.test = _SimpleCAPTDataset(_test, path)


class _SimpleCAPTDataPreprocessor(Dataset):

    def __init__(self, datalist, path, mode, transform=None):
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
        
        self.mode = {
            1: {
                "AdaptiveThreshold" : 145,
                "MorphologyEx" : 2
                },
            2: {
                "AdaptiveThreshold" : 145,
                "MorphologyEx" : 3
               },
            3: {
                "AdaptiveThreshold" : 215,
                "MorphologyEx" : 2
               },
            4: {
                "AdaptiveThreshold" : 215,
                "MorphologyEx" : 3
               }
        }

        self.modeSet = mode # list such as [1, 2, 4]
        if (max(self.modeSet) > 4):
            raise ValueError("Mode exceeding 4")


    def __len__(self):
        # return length
        return len(self.imgList)

    def __getitem__(self, idx):

        _mode = np.random.choice(self.modeSet)
        _img = Image.open(os.path.join(self.path, self.imgList[idx]))
        _img = _img.convert('L')
        _p_img = cv2.imread(os.path.join(self.path, self.imgList[idx]), cv2.IMREAD_GRAYSCALE)
        _p_img = self.process_img(_p_img, _mode)
        _p_img = Image.fromarray(_p_img)
        _label = torch.zeros(len(ALPHANUMERIC)*SIMPLECAPTCHALEN)
        if (self.transform is not None):
            _img = self.transform(_img)
            _p_img = self.transform(_p_img)
        for c, i in zip(self.imgList[idx], range(SIMPLECAPTCHALEN)):
            _label[ALPHANUMERIC.index(c) + i*len(ALPHANUMERIC)] = 1
        
        return _img, _p_img, _label

    def process_img(self, img, mode):
        # mode: 1, 2, 3, or 4

        _img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     self.mode[1]["AdaptiveThreshold"], 0)
        _img = cv2.morphologyEx(_img, cv2.MORPH_CLOSE,
                                np.ones((5, self.mode[1]["MorphologyEx"]), np.uint8))
        _img = cv2.dilate(_img, np.ones((2, 2), np.uint8), iterations=1)
        _img = cv2.GaussianBlur(_img, (1, 1), 0)
        return _img

class SimpleCAPTCHAPreProcessor():
    
    def __init__(self, path, mode, transform=None):
        _file_list = (os.listdir(path))
        _file_list.remove(".DS_Store")
        np.random.seed(0)
        self.train = _SimpleCAPTDataPreprocessor(_file_list, path, mode, transform)