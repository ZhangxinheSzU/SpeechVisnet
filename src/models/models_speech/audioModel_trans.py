import torch
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Sigmoid, Conv2d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, LeakyReLU, AdaptiveMaxPool1d

from transformers import AudioModel 

from src.config import cfg

class Reshape(nn.Module):
    def forward(self, x):
        #print(x)
        batch_size = x.size()[0]
        feature_len = x.size()[-1]
        x = x.view((batch_size, -1, feature_len))
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 3, 3)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = Sequential( 
            Conv2d(1, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros' ),
            BatchNorm2d(128),
            ReLU(),  # 128x48x48
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x24x24
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x12x12
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x6x6
            Dropout(.25),
            Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),  # 128x3x3
            Dropout(.25),
            #Reshape(),
            #AdaptiveMaxPool1d(1),
            #Flatten(),
        )
        

        self.fc_audio = Sequential(
            Reshape(),
            AdaptiveMaxPool1d(1),
            Flatten(),
            Linear(384, cfg.TAE.dim_z, bias=False),
            Dropout(0.25),
        )

    def forward(self, x):
        #print('audio input', x.size())
        #for i in range(len(self.audio_encoder)):
        #    x = self.audio_encoder[i](x)
        #    print('layer %d: '%i, x.size())
        #z = x
        
        z = self.audio_encoder(x)
        speech_embedding = self.fc_audio(z)
        #print('speech_embedding', speech_embedding.size(), z.size())
        speech_states = Reshape()(z)

        speech_states = self.fc_audio[3](speech_states.transpose(1, 2))

        return z, speech_embedding, speech_states


class AudioDecoder(nn.Module):
    def __init__(self):
        super(AudioDecoder, self).__init__()

        self.audio_decoder = Sequential(
            #UnFlatten(),
            #Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            Dropout(.25),
            ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(128),
            ReLU(),
            ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, padding_mode='zeros'),
            BatchNorm2d(1),
            Sigmoid(),
        )

    def forward(self, z):
        return self.audio_decoder(z)


