import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from model.encoder import Encoder
from model.encoder import CNN_PRESET
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SimpleCAPTCHA
from dataset.dataset import LargeCAPTCHA
from dataset.dataset import SIMPLECAPTCHALEN
from dataset.dataset import ALPHANUMERIC
from dataset.dataset import ALPHANUMERIC_UPPERLOWER

class EncoderClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()

        if(params['load_encoder']):
            self.encoder = self._load_encoder(params)
        else:
            self.encoder = Encoder(params)

        _classifier = torchvision.models.resnet18(pretrained = False)
        _classifier.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),
                                            padding=(3,3),bias=False)
        _classifier.fc = nn.Linear(512, SIMPLECAPTCHALEN * len(ALPHANUMERIC_UPPERLOWER))

        self.classifier = _classifier

    def forward(self, x):
        _enc = self.encoder(x)
        y = self.classifier(_enc)
        return y

    def _load_encoder(self, params):
        _path = os.path.join(params['encoder_path'], "model.pt")
        _model = torch.load(_path)
        return _model

    def freeze_encoder(self):
        for _param in self.encoder.parameters():
            _param.require_grad = False

    def unfreeze_encoder(self):
        for _param in self.encoder.parameters():
            _param.require_grad = True


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
    _data_path = os.path.join(params["data_path"], params['dataset'])
    os.makedirs(_model_save_path, exist_ok = True)

    # model
    _Model = EncoderClassifier(params)
    _Model.freeze_encoder()
    #_Loss = nn.CrossEntropyLoss()
    _Loss = nn.MultiLabelSoftMarginLoss()
    #_Loss = nn.KLDivLoss()
    _Optim = torch.optim.Adam(_Model.parameters(), lr=params['lr'])
    if USE_CUDA:
        _Model.cuda()

    # transform
    _transform = [torchvision.transforms.Compose([
                        #torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ]),
                    torchvision.transforms.Compose([
                        #torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ]),
                    torchvision.transforms.Compose([
                        #torchvision.transforms.Resize([224, 224]),
                        torchvision.transforms.ToTensor(),
                    ])]

    # dataset
    _captdata = LargeCAPTCHA(_data_path, (0.8, 0.1, 0.1), transform =_transform)
    _train_loader = DataLoader(_captdata.train, batch_size=params['batch_size'])
    _valid_loader = DataLoader(_captdata.valid, batch_size=params['batch_size'])
    _test_loader = DataLoader(_captdata.test, batch_size=params['batch_size'])

    accuracy = _accuracy_large
    # record
    _record = np.zeros((params['epoch'], 4)) # train_loss, # train_acc, #validation_acc, #test_acc

    for e in range(params['epoch']):
        if (e == int(params['epoch']/2)):
            _Model.unfreeze_encoder()
        _avg_loss = 0.0
        _avg_val_acc = 0.0
        _avg_tr_acc = 0.0
        _avg_ts_acc = 0.0
        _pbar = tqdm(total=len(_train_loader))
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
            _avg_tr_acc += accuracy(_pred, _label)
            _pbar.update()
        _pbar.close()

        for _, (_img, _label) in enumerate(_valid_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()

            _pred = _Model(_img)
            _avg_val_acc += accuracy(_pred, _label)

        for _, (_img, _label) in enumerate(_test_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _label = _label.cuda()

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
        'data_path' : './dataset',
        'dataset': 'CAPTCHA_LARGE',
        'save_path':  './model/proposed/',
        'load_encoder': True,
        'encoder_path': './model/saved_encoder_processors/CAPTCHA_LARGE/160',
        'layer_config' : [-1, 0, -1, 0, 0, 1, 0, 1],
        'cnn_config' : CNN_PRESET[0],
        'device': 'cuda:0'
    }
    run(_params)