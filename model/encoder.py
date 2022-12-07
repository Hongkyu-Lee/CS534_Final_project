import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.in_planes = 64
        self.layers = []
        self._layer_config = params['layer_config']
        self._upsample_config = params['cnn_config']['upsample']
        self._downsample_config = params['cnn_config']['downsample']
        self._identity_config = params['cnn_config']['identity']

        self.in_layer = self._in_block()

        for c in self._layer_config:
            self.layers.append(self._make_block(c))
        self.layers = nn.Sequential(*self.layers)

        self.out_layer = self._out_block()

    def forward(self, x):
        
        y = self.in_layer(x)
        y = self.layers(y)
        y = self.out_layer(y)
        y = torch.tanh(y)

        return y

    def _in_block(self):
        layers = []
        layers.append(nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(self.in_planes))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _out_block(self):
        layers = []
        layers.append(nn.Upsample(size=(50, 200)))
        layers.append(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)
    
    def _make_block(self, factor = 0):
        layers = []

        if factor == 0: # identity
            layers.append(nn.Conv2d(64, 64, kernel_size=self._identity_config['kernel_size'],
                                            stride=self._identity_config['stride'],
                                            padding=self._identity_config['padding'], bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        elif factor < 0: # downsample
            layers.append(nn.Conv2d(64, 64, kernel_size=self._downsample_config['kernel_size'],
                                            stride=self._downsample_config['stride'],
                                            padding=self._downsample_config['padding'], bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        elif factor > 0: # upsample
            layers.append(nn.Upsample(scale_factor=2.0))
            layers.append(nn.Conv2d(64, 64, kernel_size=self._upsample_config['kernel_size'],
                                            stride=self._upsample_config['stride'],
                                            padding=self._upsample_config['padding'], bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)



CNN_PRESET = (
    {
        "identity" : {
            "kernel_size" : 3,
            "stride" : 1,
            "padding" : 1
        },
        "upsample" : {
            "kernel_size" : 3,
            "stride" : 1,
            "padding" : 1
        },
        "downsample": {
            "kernel_size" : 3,
            "stride" : 2,
            "padding" : 1
        }
    },
    {
        "identity" : {
            "kernel_size" : 5,
            "stride" : 1,
            "padding" : 2
        },
        "upsample" : {
            "kernel_size" : 5,
            "stride" : 1,
            "padding" : 2
        },
        "downsample": {
            "kernel_size" : 5,
            "stride" : 2,
            "padding" : 2
        }  
    },
    {
        "identity" : {
            "kernel_size" : 7,
            "stride" : 1,
            "padding" : 3
        },
        "upsample" : {
            "kernel_size" : 7,
            "stride" : 1,
            "padding" : 3
        },
        "downsample": {
            "kernel_size" : 7,
            "stride" : 2,
            "padding" : 3
        }  
    }
)