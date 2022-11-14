import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from dataset.dataset import SimpleCAPTCHA
from dataset.dataset import SIMPLECAPTCHALEN
from dataset.dataset import ALPHANUMERIC



def accuracy_(pred, true):
    '''
    TODO: move this function to .utils
    '''
    
    _subtract = torch.square(torch.ceil(pred)-true)
    _nonzero = torch.count_nonzero(torch.sum(_subtract, 1))
    print(len(pred), _nonzero)
    return len(pred) - _nonzero

def accuracy(pred, true):
    acc = 0.0
    _ans = np.zeros((len(pred), SIMPLECAPTCHALEN))
    _tru = np.zeros((len(pred), SIMPLECAPTCHALEN))
    for _a, _p, _t, _l in zip(_ans, pred, _tru, true):
        for _i in range(SIMPLECAPTCHALEN):
            _a[_i] = torch.argmax(_p[_i*len(ALPHANUMERIC):(_i+1)*len(ALPHANUMERIC)])
            _t[_i] = torch.argmax(_l[_i*len(ALPHANUMERIC):(_i+1)*len(ALPHANUMERIC)])
    for _a, _t in zip(_ans, _tru):
        for _i in range(SIMPLECAPTCHALEN):
            if(_a[_i] == _t[_i]):
                acc += 1
            
    return acc / (SIMPLECAPTCHALEN*len(pred))

    


def run(params):

    '''
    Vanila CNN training
    model : resnet 18
    
    '''
    # cuda
    if torch.cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False

    # model
    _Model = torchvision.models.resnet18(pretrained = False)
    _Model.fc = nn.Linear(512, SIMPLECAPTCHALEN * len(ALPHANUMERIC))
    #_Loss = nn.CrossEntropyLoss()
    _Loss = nn.MultiLabelSoftMarginLoss()
    #_Loss = nn.KLDivLoss()
    _Optim = torch.optim.Adam(_Model.parameters(), lr=1e-4)
    if USE_CUDA:
        _Model.cuda()

    # transform
    _transform = [torchvision.transforms.Compose([
                        torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ]),
                    torchvision.transforms.Compose([
                        torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ]),
                    torchvision.transforms.Compose([
                        torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ])]

    # dataset
    _captdata = SimpleCAPTCHA((0.8, 0.1, 0.1), seed=0, path=params['path'], transform =_transform)
    _train_loader = DataLoader(_captdata.train, batch_size=params['batch_size'])
    _valid_loader = DataLoader(_captdata.valid, batch_size=params['batch_size'])
    _test_loader = DataLoader(_captdata.test, batch_size=params['batch_size'])

    for e in range(params['epoch']):
        _avg_loss = 0.0
        _avg_acc = 0.0
        for _, (_img, _label) in enumerate(_train_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()
            
            _pred = _Model(_img)
            _loss = _Loss(_pred, _label)
            _Optim.zero_grad()
            _loss.backward()
            _Optim.step()
            _avg_loss += _loss.item()

        for _, (_img, _label) in enumerate(_valid_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()

            _pred = _Model(_img)
            _avg_acc += accuracy(_pred, _label)
        print(f"Epoch: {e}, train loss: {_avg_loss/len(_train_loader)} validation accuracy: {_avg_acc/len(_valid_loader)}")
        

if __name__ == "__main__":
    _params = {
        'epoch': 100,
        'batch_size': 64,
        'path' : './dataset/CAPTCHA_SIMPLE'
    }
    run(_params)
