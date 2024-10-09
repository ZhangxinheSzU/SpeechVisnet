#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/13
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils.dataset import Batch
from src.config import cfg
from src.models.common_modules import Attention, Transformer
from src.models.models_speech import SQLDecoder, audioModel, textModel
from src.models.models_match import SchemaModel
import torchaudio.transforms as T
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Model

from src.config import cfg

class SpeechSQLNet(nn.Module):
    def __init__(self, grammar, vocab,glove_embed,wav2vec2, use_gcn = False):
        super(SpeechSQLNet, self).__init__()
        self.grammar = grammar
        self.word_emb = vocab
        self.glove_emb = glove_embed
        # self.processor = processor
        # self.speech_encoder =  Wav2Vec2Model.from_pretrained('/home/zhoujh/speech2SQL/pretrain/data',cache_dir='/home/zhoujh/speech2SQL/pretrain/data')
        self.speech_encoder =wav2vec2
        self.word_encoder = textModel.TextEncoder(self.word_emb, self.glove_emb,cfg.TAE)
        self.encoder_lstm = nn.LSTM(300,300 // 2, bidirectional=True,
                                    batch_first=True)
        self.use_gcn = use_gcn
        if self.use_gcn:
            self.gcn = SchemaModel.MultiLayerGCN(cfg.SCHEMA_GCN.num_layers)

        self.encoder  = Transformer.TransformerEncoder(cfg.TRANSFORMER.max_seq_len,cfg.cuda, cfg.TRANSFORMER.num_layers, cfg.TRANSFORMER.hidden_size, 
                                                                  cfg.TRANSFORMER.n_heads, cfg.TRANSFORMER.dim_feedforward, cfg.TRANSFORMER.dropout)
        self.decoder = SQLDecoder.SQLDecoder(self.grammar)


    def get_decoder_inputs(self, examples):
        batch = Batch(examples, self.grammar, self.word_emb.word2idx, cuda = cfg.cuda, word_encoder = self.word_encoder)
        
        # speech_outputs = self.speech_encoder(batch.src_speeches)['last_hidden_state'].squeeze(dim=0)
        speech_outputs = self.speech_encoder(batch.src_sents)

        if self.use_gcn:
            schema_outputs = self.gcn(batch.schema_sents_embedding, batch.schema_adj)
        else:
            schema_outputs = batch.schema_sents_embedding
        schema_outputs = nn.utils.rnn.pad_sequence(schema_outputs, batch_first = True)
        decoder_speeches_len = [e.shape[0] for e in speech_outputs]
        # speech_outputs = nn.utils.rnn.pad_sequence(speech_outputs, batch_first = True)
        encoder_inputs = torch.cat([speech_outputs, schema_outputs], axis = 1)
        
        # 
        attn_mask = None 
        encoder_inputs_len = None
        attn_mask = batch.padding_mask(decoder_speeches_len,batch.schema_sents_len)
        # 啥意思啊？ 为什么要np.sum
        # encoder_inputs_len = np.sum([batch.decoder_speeches_len, batch.schema_sents_len], axis = 0)
        encoder_inputs_len = np.sum([decoder_speeches_len, batch.schema_sents_len], axis = 0)
        pos = True if encoder_inputs_len is not None else False
        # 有个问题：schema和音频长度不一定对应  老师的做法音频长度是都变成10？
        encoder_outputs, encoder_attentions = self.encoder(encoder_inputs, inputs_len = encoder_inputs_len, pos = pos, attn_mask = attn_mask)

        schema_len = schema_outputs.size()[1]
        schema_outputs = encoder_outputs[:, -schema_len:] 
        encoder_outputs = encoder_outputs[:, :-schema_len] 

        return batch, encoder_outputs, schema_outputs


    def forward(self, examples):
        batch, encoder_outputs, schema_outputs = self.get_decoder_inputs(examples)
        decoder_outputs = self.decoder(batch, encoder_outputs, schema_outputs)
        return decoder_outputs, 0
    
    def parse(self, examples, beam_size):
        batch, encoder_outputs, schema_outputs = self.get_decoder_inputs(examples)
        match_outputs = None
        #match_outputs = self.get_match_outputs(examples, speech_embeddings)
        decoder_outputs = self.decoder.parse(batch, encoder_outputs, schema_outputs, beam_size, match_outputs)
        return decoder_outputs

    def get_match_outputs(self, examples, speech_embeddings):
        '''
            evaluate the results when given the gold label which marks that if one item has been choosed to appear in the SQL,
        '''
        preds = []
        for i in range(len(speech_embeddings)):
            if examples is not None:
                match_label = examples[i].label_match + 1e-5
                pred = torch.from_numpy(np.array(match_label, dtype = np.float32)).cuda()
                pred = pred.unsqueeze(-1)
            preds.append(pred)
        return preds

