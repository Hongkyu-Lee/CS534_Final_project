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

ALPHANUMERIC_UPPERLOWER = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 
                           'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                           'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                           'Y', 'Z')

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
    
    def __init__(self, path, split, transform=None):
        np.random.seed(0)
        _file_list = (os.listdir(path))
        if ".DS_Store" in _file_list:
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
    
    def __init__(self, path, split, mode, transform=None):
        
        _file_list = (os.listdir(path))
        _file_list.remove(".DS_Store")
        
        if split is None: 
            self.train = _SimpleCAPTDataPreprocessor(_file_list, path, mode, 
                                                      transform)
        
        else:
            _len = len(_file_list)
            _rand_file_list = np.random.choice(_file_list, size=_len, replace=False)
            _train  = _rand_file_list[0:int(_len*split[0])]
            _valid  = _rand_file_list[int(_len*split[0]):int(_len*split[0])+int(_len*split[1])]
            _test   = _rand_file_list[int(_len*split[0])+int(_len*split[1]):]
            self.train = _SimpleCAPTDataPreprocessor(_train, path, mode, transform[0])
            self.valid = _SimpleCAPTDataPreprocessor(_valid, path, mode, transform[1])
            self.test = _SimpleCAPTDataPreprocessor(_test, path, mode, transform[2])
            
class _LargeCAPTDataset(Dataset):

    def __init__(self, datalist, path, transform=None):

        self.path = path
        self.transform = transform
        np.random.seed(0)

        self.imgList = datalist

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        _img = Image.open(os.path.join(self.path, self.imgList[idx]))
        _img = _img.convert('L')
        _label = torch.zeros(len(ALPHANUMERIC)*SIMPLECAPTCHALEN)
        if (self.transform is not None):
            _img = self.transform(_img)
        for c, i in zip(self.imgList[idx].lower(), range(SIMPLECAPTCHALEN)):
            _label[ALPHANUMERIC.index(c) +
                   i*len(ALPHANUMERIC)] = 1
        
        if(torch.mean(_img) > 0.5):
            _img = 1-_img

        return _img, _label


class LargeCAPTCHA():

    def __init__(self, path, split, transform=None):
        np.random.seed(0)
        _file_list = (os.listdir(path))
        if ".DS_Store" in _file_list:
            _file_list.remove(".DS_Store")
        _file_list = _file_list[:10000] # first 10,000 images 
        _len = len(_file_list)
        _rand_file_list = np.random.choice(_file_list, size=_len, replace=False)
        _train  = _rand_file_list[0:int(_len*split[0])]
        _valid  = _rand_file_list[int(_len*split[0]):int(_len*split[0])+int(_len*split[1])]
        _test   = _rand_file_list[int(_len*split[0])+int(_len*split[1]):]

        self.train = _LargeCAPTDataset(_train, path, transform[0])
        self.valid = _LargeCAPTDataset(_valid, path, transform[1])
        self.test = _LargeCAPTDataset(_test, path, transform[2])

class _LargeCAPTDataPreprocessor(Dataset):
    def __init__(self, datalist, path, mode, transform=None):

        self.path = path
        self.transform = transform

        np.random.seed(0)

        self.imgList = datalist

        self.mode = mode

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        _img = Image.open(os.path.join(self.path, self.imgList[idx]))
        _img = _img.convert('L')
        _p_img = cv2.imread(os.path.join(self.path, self.imgList[idx]), cv2.IMREAD_GRAYSCALE)
        _p_img = self.process_img(_p_img)
        _p_img = Image.fromarray(_p_img)
        _label = torch.zeros(len(ALPHANUMERIC)*SIMPLECAPTCHALEN)
        if (self.transform is not None):
            _img = self.transform(_img)
            _p_img = self.transform(_p_img)
        for c, i in zip(self.imgList[idx].lower(), range(SIMPLECAPTCHALEN)):
            _label[ALPHANUMERIC.index(c) + i*len(ALPHANUMERIC)] = 1
        
        return _img, _p_img, _label


    def process_img(self, img):
        _img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 215, 1) #stricter
        _img = self._conditional_invert(_img)
        _img = self._rmv_horiz(_img)
        _img = cv2.GaussianBlur(_img, (1,1), 1)

        return _img

    def _conditional_invert(self, img):
        if np.mean(img) >= 128:
            img = cv2.bitwise_not(img)
        return(img)

    def _rmv_horiz(self, img): #thicker lines
        #remove horizontal
        if self.mode ==1:
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
            detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(img, [c], -1, (255,255,255), 2)

            # Repair image
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
            result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        elif self.mode == 2: #thinner
            #remove horizontal
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
            detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(img, [c], -1, (255,255,255), 2)

            # Repair image
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
            result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

        elif self.mode == 3: # both
            if(random.random() > 0.5):
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
                detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
                cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(img, [c], -1, (255,255,255), 2)

                # Repair image
                repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
                result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
            else:
                #remove horizontal
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
                detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
                cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(img, [c], -1, (255,255,255), 2)

                # Repair image
                repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
                result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        
        return result

    # def rmv_horiz(img): #thicker lines
    #     #remove horizontal
    #     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    #     detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
    #     cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #     for c in cnts:
    #         cv2.drawContours(img, [c], -1, (255,255,255), 2)

    #     # Repair image
    #     repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
    #     result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    #     return result, detected_lines

    # def rmv_horiz1(img): #thinner
    #     #remove horizontal
    #     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    #     detected_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=10)
    #     cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #     for c in cnts:
    #         cv2.drawContours(img, [c], -1, (255,255,255), 2)

    #     # Repair image
    #     repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    #     result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    #     return result, detected_lines

class LargeCAPTCHAPreProcessor():

    def __init__(self, path, split, mode, transform=None):
        
        _file_list = (os.listdir(path))
        if ".DS_Store" in _file_list:
            _file_list.remove(".DS_Store")
        _file_list = _file_list[:10000] # first 10,000 images 
        
        if split is None: 
            self.train = _LargeCAPTDataPreprocessor(_file_list, path, mode, 
                                                      transform)
        
        else:
            _len = len(_file_list)
            _rand_file_list = np.random.choice(_file_list, size=_len, replace=False)
            _train  = _rand_file_list[0:int(_len*split[0])]
            _valid  = _rand_file_list[int(_len*split[0]):int(_len*split[0])+int(_len*split[1])]
            _test   = _rand_file_list[int(_len*split[0])+int(_len*split[1]):]
            self.train = _LargeCAPTDataPreprocessor(_train, path, mode, transform[0])
            self.valid = _LargeCAPTDataPreprocessor(_valid, path, mode, transform[1])
            self.test = _LargeCAPTDataPreprocessor(_test, path, mode, transform[2])