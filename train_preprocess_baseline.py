import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from dataset.dataset import SimpleCAPTCHAPreProcessor
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

    _model_save_path = os.path.join(params["save_path"],
                                    str(params['denoise_mode'][0]),
                                    str(time.time()))
    os.makedirs(_model_save_path, exist_ok = True)

    # model
    _Model = torchvision.models.resnet18(pretrained = False)
    _Model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
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
    _captdata = SimpleCAPTCHAPreProcessor(path=params['path'], split=(0.8, 0.1, 0.1),
                                          mode=params['denoise_mode'], transform =_transform)
    _train_loader = DataLoader(_captdata.train, batch_size=params['batch_size'])
    _valid_loader = DataLoader(_captdata.valid, batch_size=params['batch_size'])
    _test_loader = DataLoader(_captdata.test, batch_size=params['batch_size'])

    for e in range(params['epoch']):
        _avg_loss = 0.0
        _avg_acc = 0.0
        _avg_tr_acc = 0.0
        for _, (_img, _p_img, _label) in enumerate(_train_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()
            
            _pred = _Model(_img)
            _loss = _Loss(_pred, _label)
            _Optim.zero_grad()
            _loss.backward()
            _Optim.step()
            _avg_loss += _loss.item()

            _avg_tr_acc += accuracy(_pred, _label)

        for _, (_img, _p_img, _label) in enumerate(_valid_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()

            _pred = _Model(_img)
            _avg_acc += accuracy(_pred, _label)
        print(f"Epoch: {e}, train loss: {_avg_loss/len(_train_loader)} validation accuracy: {_avg_acc/len(_valid_loader)}")
        np.savetxt(os.path.join(_model_save_path, "record.csv"), _record, delimiter=',')
        
    # save model
    if USE_CUDA:
        _Model.cpu()

    torch.save(_Model, os.path.join(_model_save_path, "model.pt"))
    _Model_script = torch.jit.script(_Model)
    _Model_script.save(os.path.join(_model_save_path, "model_jit_scr.pt"))
    torch.save({
                'loss' : _avg_loss/len(_train_loader),
                'train_acc' : _avg_tr_acc / len(_train_loader),
                'val_acc': _avg_acc/len(_valid_loader),
                'state_dict': _Model.state_dict()
                }, os.path.join(_model_save_path, "model_state_dict.pt"))


if __name__ == "__main__":
    _params = {
        'epoch': 200,
        'batch_size': 64,
        'lr': 1e-4,
        'path' : './dataset/CAPTCHA_SIMPLE',
        'denoise_mode': [3],
        'save_path':  './model/baseline/preprocessing'
    }
    run(_params)
