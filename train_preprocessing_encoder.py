import os
import json
import time
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.encoder import CNN_PRESET
import torchvision
from torch.utils.data import DataLoader

from dataset.dataset import SimpleCAPTCHAPreProcessor
from dataset.dataset import SIMPLECAPTCHALEN
from dataset.dataset import ALPHANUMERIC



def train_encoder(params):

    encoder_save_path = os.path.join(params['encoder_save_path'], str(time.time()))
    os.makedirs(encoder_save_path, exist_ok=True)

    # cuda
    if torch.cuda.is_available():
        USE_CUDA = True
    else:
        USE_CUDA = False

    _Model = Encoder(params)
    _Loss = nn.MSELoss()
    _Optim = torch.optim.Adam(_Model.parameters(), lr=params['lr'])

    if USE_CUDA:
        _Model.cuda()

    # transform 
    _transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                    ])

    # dataset
    if params['dataset'] == 'CAPTCHA_SIMPLE':
        _datapath = os.path.join(params['datapath'], params['dataset'])
        _data = SimpleCAPTCHAPreProcessor(path=_datapath, split=None, mode=params['denoise_mode'],
                                          transform=_transform)
    elif params['dataset'] == 'CAPTCHA_LARGE':
        # TBA
        exit()

    _train_loader = DataLoader(_data.train, batch_size=params['batch_size'])

    for e in range(params['epoch']):
        _avg_loss = 0.0
        _avg_acc = 0.0
        for _, (_img, _p_img, _label) in enumerate(_train_loader):
            if USE_CUDA:
                _img = _img.cuda()
                _p_img = _p_img.cuda()
            
            _pred = _Model(_img)
            _loss = _Loss(_pred, _p_img)
            _Optim.zero_grad()
            _loss.backward()
            _Optim.step()
            _avg_loss += _loss.item()

        print(f"Epoch: {e}, train loss: {_avg_loss/len(_train_loader)}")    

    # save model
    if USE_CUDA:
        _Model.cpu()
    
    with open(os.path.join(encoder_save_path, "params.txt"), 'w') as _dict_file:
        _dict_file.write(json.dumps(params))

    torch.save(_Model, os.path.join(encoder_save_path, "model.pt"))
    _Model_script = torch.jit.script(_Model)
    _Model_script.save(os.path.join(encoder_save_path, "model_jit_scr.pt"))
    torch.save({
                'loss' : _avg_loss/len(_train_loader),
                'state_dict': _Model.state_dict()
                }, os.path.join(encoder_save_path, "model_state_dict.pt"))

if __name__ == "__main__":

    _params = {
        'epoch': 2000,
        'batch_size': 64,
        'lr' : 1e-4,
        'layer_config' : [-1, 0, -1, 0, 0, 1, 0, 1],
        'cnn_config' : CNN_PRESET[0],
        'denoise_mode' : [1, 3],
        'encoder_save_path' : './model/saved_encoder_processors',
        'datapath' : './dataset/',
        'dataset' : 'CAPTCHA_SIMPLE'
    }

    train_encoder(_params)