#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/26
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import traceback
import torch
import torch.optim as optim
import logging
from tqdm import tqdm
from src.config import cfg, args
from src.utils import data_loader_both
from src.rule import semQL
from src.utils import vocab
from src.utils.vocab import Vocab
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from src.models.models_speech.SpeechSQLNet_both import SpeechSQLNet

import time
import copy

from src.trainer.trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] ='4'

type = 'pretrain'



def load_pretrain(model):
    model_path = '/home/zhoujh/speech2SQL/saved_model/augment_split/speech2sql/end_model.model' # speech-item model with 
    logging.info('Load pretrained parameters from: {}'.format(model_path))
    
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model



def load_word_emb(file_name, use_small=False):
    print ('Loading word embedding from %s'%file_name)
    ret = {}
    with open(file_name) as inf:
        for idx, line in enumerate(inf):
            if (use_small and idx >= 500):
                break
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x:float(x), info[1:])))
    if '<unk>' not in ret:
        ret['<unk>'] = np.random.rand(300)
    return ret

def get_params_group(model, onlyMatch = False):
    fast_params = []
    slow_params = []
    for name, params in model.named_parameters():
        if 'speech_encoder' in name or 'word_encoder' in name:
            #if 'embed' not in name:
            slow_params.append(params)
        else: 
            if onlyMatch:
                if 'match_model' in name:
                    fast_params.append(params)
            else:
                fast_params.append(params)

    return fast_params, slow_params

def get_params_num(model):
    params_num = sum(param.numel() for param in model.parameters())
    params_speech_encoder = sum(param.numel() for param in model.speech_encoder.parameters())
    params_word_encoder = sum(param.numel() for param in model.word_encoder.parameters())
    params_encoder = sum(param.numel() for param in model.encoder.parameters())
    params_decoder = sum(param.numel() for param in model.decoder.parameters())
    logging.info('total params: {}. speech_encoder params: {}. text_encoder params: {}. encoder params: {}, decoder params: {}'.format(
        params_num, params_speech_encoder, params_word_encoder, params_encoder, params_decoder))
    print('total params: {}. speech_encoder params: {}. text_encoder params: {}. encoder params: {}, decoder params: {}'.format(
        params_num, params_speech_encoder, params_word_encoder, params_encoder, params_decoder))


def train():
    onlyMatch = False   
    dev_batch_size = 100 if onlyMatch else 1
    
    
    dev_data = data_loader_both.SpeechSQLDataset(type,'/home/zhoujh/speech2SQL/data/nvbench/New_Test2_semQL.json', cfg.dev_speech_path, cfg.table_path,  normalize=True, ifeval = True,mode='new_test')
    dev_loader = data_loader_both.SpeechSQLDataLoader(dev_data, num_workers=cfg.num_workers, shuffle = False, batch_size = dev_batch_size)
   


    # initialize model
    grammar = semQL.Grammar()
    
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng",device=device)
    t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",device=device,
                                            tokenizer="text_sonar_basic_encoder")
    
    
    model = SpeechSQLNet(type, grammar,s2vec_model,t2vec_model)#, model_sw)
    if cfg.cuda: 
        model.cuda()
    
    eval_load_model = '/home/zhoujh/speech2SQL/saved_model/checkpoint/ckpt_best_speech_one_170.pth'
    continue_train = True
    epoch = 220
    if continue_train:
        
        path_checkpoint='/home/zhoujh/speech2SQL/saved_model/checkpoint/ckpt_best_pretrain_super_20.pth'
        checkpoint = torch.load(path_checkpoint,map_location=torch.device('cpu'))  # 加载断点

        model.load_state_dict(checkpoint['net'],strict=False)  # 加载模型可学习参数
        
        start_epoch = checkpoint['epoch']+1  # 设置开始的epoch
    

    trainer = Trainer()
    start = time.time()
    
    with torch.no_grad():
        json_datas, sketch_acc, acc,sum_time = trainer.val_step(model, dev_loader, beam_size = cfg.beam_size)
    print('numbers: {}'.format(len(dev_loader)))
    print('avg_time: {}'.format(sum_time/len(dev_loader)))
    print('dev time used: ', time.time() - start)
    print('Dev Sketch Acc: %f, Acc: %f' % (sketch_acc, acc))

if __name__ == '__main__':
    train()
