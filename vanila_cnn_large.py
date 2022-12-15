import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SimpleCAPTCHA
from dataset.dataset import LargeCAPTCHA
from dataset.dataset import SIMPLECAPTCHALEN
from dataset.dataset import ALPHANUMERIC
from dataset.dataset import ALPHANUMERIC_UPPERLOWER

def _accuracy_simple(pred, true):
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

def _accuracy_large(pred, true):
    acc = 0.0
    _ans = np.zeros((len(pred), SIMPLECAPTCHALEN))
    _tru = np.zeros((len(pred), SIMPLECAPTCHALEN))
    for _a, _p, _t, _l in zip(_ans, pred, _tru, true):
        for _i in range(SIMPLECAPTCHALEN):
            _a[_i] = torch.argmax(_p[_i*len(ALPHANUMERIC_UPPERLOWER):(_i+1)*len(ALPHANUMERIC_UPPERLOWER)])
            _t[_i] = torch.argmax(_l[_i*len(ALPHANUMERIC_UPPERLOWER):(_i+1)*len(ALPHANUMERIC_UPPERLOWER)])
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

    _model_save_path = os.path.join(params["save_path"], params['dataset'], str(time.time()))
    os.makedirs(_model_save_path, exist_ok = True)
    device = params['device']

    # model
    _Model = torchvision.models.resnet18(pretrained = False)
    _Model.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    if params['dataset'] == "CAPTCHA_SIMPLE":
        _Model.fc = nn.Linear(512, SIMPLECAPTCHALEN * len(ALPHANUMERIC))
    elif params['dataset'] == "CAPTCHA_LARGE":
        # _Model.fc = nn.Linear(512, SIMPLECAPTCHALEN * len(ALPHANUMERIC_UPPERLOWER))
        _Model.fc = nn.Linear(512, SIMPLECAPTCHALEN * len(ALPHANUMERIC))
    #_Loss = nn.CrossEntropyLoss()
    _Loss = nn.MultiLabelSoftMarginLoss()
    #_Loss = nn.KLDivLoss()
    _Optim = torch.optim.Adam(_Model.parameters(), lr=params['lr'])
    if USE_CUDA:
        _Model.to(device)

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
    _data_path = os.path.join(params['datapath'], params['dataset'])
    if params['dataset'] == "CAPTCHA_SIMPLE":
        _captdata = SimpleCAPTCHA(_data_path, (0.8, 0.1, 0.1), transform =_transform)
        _train_loader = DataLoader(_captdata.train, batch_size=params['batch_size'])
        _valid_loader = DataLoader(_captdata.valid, batch_size=params['batch_size'])
        _test_loader = DataLoader(_captdata.test, batch_size=params['batch_size'])
        accuracy = _accuracy_simple

    elif params['dataset'] == "CAPTCHA_LARGE":
        _captdata = LargeCAPTCHA(_data_path, (0.8, 0.1, 0.1), transform =_transform)
        _train_loader = DataLoader(_captdata.train, batch_size=params['batch_size'])
        _valid_loader = DataLoader(_captdata.valid, batch_size=params['batch_size'])
        _test_loader = DataLoader(_captdata.test, batch_size=params['batch_size'])
        # accuracy = _accuracy_large
        accuracy = _accuracy_simple

    # record
    _record = np.zeros((params['epoch'], 4)) # train_loss, # train_acc, #validation_acc, #test_acc

    # Training begin
    for e in range(params['epoch']):
        _avg_loss = 0.0
        _avg_val_acc = 0.0
        _avg_tr_acc = 0.0
        _avg_ts_acc = 0.0
        _pbar = tqdm(total=len(_train_loader))
        for _, (_img, _label) in enumerate(_train_loader):
            if USE_CUDA:
                _img = _img.to(device)
                _label = _label.to(device)
            
            _pred = _Model(_img)
            _loss = _Loss(_pred, _label)
            _Optim.zero_grad()
            _loss.backward()
            _Optim.step()
            _avg_loss += _loss.item()

            _avg_tr_acc += accuracy(_pred, _label)
            _pbar.update()
        
        _pbar.close()
        for _, (_img, _label) in enumerate(_valid_loader):
            if USE_CUDA:
                _img = _img.to(device)
                _label = _label.to(device)

            _pred = _Model(_img)
            _avg_val_acc += accuracy(_pred, _label)

        for _, (_img, _label) in enumerate(_test_loader):
            if USE_CUDA:
                _img = _img.to(device)
                _label = _label.to(device)

            _pred = _Model(_img)
            _avg_ts_acc += accuracy(_pred, _label)
        

        _record[e, 0] = _avg_loss / len(_train_loader)
        _record[e, 1] = _avg_tr_acc / len(_train_loader)
        _record[e, 2] = _avg_val_acc / len(_valid_loader)
        _record[e, 3] = _avg_ts_acc / len(_test_loader)
        print(f"Epoch: {e}, train loss: {_avg_loss/len(_train_loader)} validation accuracy: {_avg_val_acc/len(_valid_loader)}")
        np.savetxt(os.path.join(_model_save_path, "record.csv"), _record, delimiter=',')
    
    # save model
    if USE_CUDA:
        _Model.cpu()

    np.savetxt(os.path.join(_model_save_path, "record.csv"), _record, delimiter=',')
    torch.save(_Model, os.path.join(_model_save_path, "model.pt"))
    _Model_script = torch.jit.script(_Model)
    _Model_script.save(os.path.join(_model_save_path, "model_jit_scr.pt"))
    torch.save({
                'loss' : _avg_loss/len(_train_loader),
                'train_acc' : _avg_tr_acc / len(_train_loader),
                'val_acc': _avg_val_acc/len(_valid_loader),
                'state_dict': _Model.state_dict()
                }, os.path.join(_model_save_path, "model_state_dict.pt"))

if __name__ == "__main__":
    _params = {
        'epoch': 100,
        'batch_size': 64,
        'lr': 1e-4,
        'datapath' : './dataset',
        'dataset': 'CAPTCHA_LARGE',
        'save_path':  './model/baseline/vanila',
        'device': 'cuda:0'
    }
    run(_params)
