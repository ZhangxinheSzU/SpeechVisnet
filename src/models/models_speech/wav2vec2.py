import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.utils.dataset import Batch
from src.config import cfg
from src.models.common_modules import Attention, Transformer
from src.models.models_speech import SQLDecoder, audioModel, textModel
from src.models.models_match import SchemaModel
import torchaudio.transforms as T

class wav2vec2Model(nn.Module):
    def __init__(self,wav2vec2):
        super(wav2vec2Model, self).__init__()
        
        # self.processor = processor
        self.wav2vec2 = wav2vec2  
        self.target_sample_rate = 16000


    def forward(self, x):        
        
        # 这个放到数据预处理部分吧
        # resampler = T.Resample(original_sample_rate, target_sample_rate)
        # resampled_audio = resampler(original_audio_input)
        # input_values = self.processor(x.cpu(), sampling_rate=self.target_sample_rate, return_tensors="pt").input_values
        # input_values = input_values.squeeze(dim=1)
        x = torch.Tensor([item.cpu().detach().numpy() for item in x])
        if cfg.cuda:
            x = x.cuda()
        x = x.squeeze(1)
        y = self.wav2vec2(x)['last_hidden_state']
        # y = y.squeeze(dim=0)
        
        return y
    
