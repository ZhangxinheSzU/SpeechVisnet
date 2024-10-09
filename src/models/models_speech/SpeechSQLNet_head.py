#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/13
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


from src.utils.dataset import Batch
from src.config import cfg
from src.models.common_modules import Attention, Transformer
from src.models.models_speech import SQLDecoder_head, audioModel, textModel
from src.models.models_match import SchemaModel
# import torchaudio.transforms as T
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Model
from src.models.models_match import  MatchModel

from src.config import cfg

class SpeechSQLNet(nn.Module):
    def __init__(self,type, grammar,s2vec,t2vec,heads,use_gcn = False, model_sw = None):
        super(SpeechSQLNet, self).__init__()
        self.type = type
        self.grammar = grammar
        # self.word_emb = vocab
        # self.glove_emb = glove_embed
        # self.processor = processor
        # self.speech_encoder =  Wav2Vec2Model.from_pretrained('/home/zhoujh/speech2SQL/pretrain/data',cache_dir='/home/zhoujh/speech2SQL/pretrain/data')
        self.speech_encoder =s2vec
        self.text_encoder = t2vec
        self.heads = heads
        # self.word_encoder = textModel.TextEncoder(self.word_emb,cfg.TAE)
        # self.encoder_lstm = nn.LSTM(300,300 // 2, bidirectional=True,
        #                             batch_first=True)
        # self.use_gcn = use_gcn
        # if self.use_gcn:
        #     self.gcn = SchemaModel.MultiLayerGCN(cfg.SCHEMA_GCN.num_layers)

        # self.encoder  = Transformer.TransformerEncoder(cfg.TRANSFORMER.max_seq_len,cfg.cuda, cfg.TRANSFORMER.num_layers, cfg.TRANSFORMER.hidden_size, 
        #                                                           cfg.TRANSFORMER.n_heads, cfg.TRANSFORMER.dim_feedforward, cfg.TRANSFORMER.dropout)
       
        self.decoder = SQLDecoder_head.SQLDecoder(self.grammar,heads)
        # self.match_model = MatchModel.MatchNet()
        # self.activation = nn.Sigmoid()
        # self.schema2speech = nn.Linear(1024, 1024)

    def get_cos_similarity(self, tensor1, tensor2):
        cos_sims = []
        for sent,schema in zip(tensor1,tensor2):
            z = torch.matmul(sent, schema.t())

            norm1 = torch.norm(sent, p = 2, dim = 1)
            norm2 = torch.norm(schema, p = 2, dim = 1)
            norm = norm1.unsqueeze(1) * norm2.unsqueeze(0)

            # L2 norm
            cos_sim = z / norm
            #cos_sim = torch.cosine_similarity(tensor1, tensor2, -1)
            cos_sims.append(cos_sim)
        cos_sims = torch.stack(cos_sims)

        return cos_sims
    
    def get_sw_embedding(self, speech_embeddings, schema_embeddings, sims):
        speech_embeddings_new = []
        
        #print('speech_embedding', speech_embeddings.size())
        #print('schema_embedding', schema_embeddings[0].size(), schema_embeddings[1].size())
        #print('sims', sims[0].size())
        for i in range(len(speech_embeddings)):
            sim = sims[i]

            sim = self.activation(sim)
            speech_embedding = speech_embeddings[i]
            schema_embedding = schema_embeddings[i]
            schema_embedding_new = torch.matmul(sim, schema_embedding) 
            schema_embedding_new = schema_embedding_new / schema_embedding.size(1)
            schema_embedding_new = self.schema2speech(schema_embedding_new)

            #speech_embedding_new = torch.cat([speech_embedding, schema_embedding], dim = -1)
            speech_embedding_new  = speech_embedding + schema_embedding_new
            #speech_embedding_new = self.speech_fc(speech_embedding_new)
            speech_embeddings_new.append(speech_embedding_new)
            #print('speech_embedding_new', speech_embedding_new.size())
        speech_embeddings_new = torch.stack(speech_embeddings_new)
        return speech_embeddings_new

    def get_decoder_inputs(self, examples):
        # batch = Batch(self.type,examples, self.grammar, self.word_emb.word2idx, cuda = cfg.cuda, word_encoder = self.word_encoder)
        batch = Batch(self.type,examples, self.grammar, cuda = cfg.cuda, word_encoder = self.text_encoder)
        # speech_outputs = self.speech_encoder(batch.src_speeches)['last_hidden_state'].squeeze(dim=0)
        # if self.type=='speech'or self.type=='speech_seperate'or self.type=='speech_one'or self.type=='speech_schema'or self.type=='speech_schemaAttention'or self.type=='speech_schema_straight':
        # tic = time.perf_counter()
        outputs = self.speech_encoder.predict(batch.src_sents)
        
            # outputs = outputs.squeeze(0)
        # else:
        #     outputs = batch.text_seqs_embedding
        #     outputs = nn.utils.rnn.pad_sequence(outputs, batch_first = True)
        # if self.use_gcn:
        #     schema_outputs = self.gcn(batch.schema_sents_embedding, batch.schema_adj)
        # else:
        #     schema_outputs = batch.schema_sents_embedding
        schema_outputs = batch.schema_sents_embedding.unsqueeze(0)
        # 没有这行就会出错： Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal tensor and use it in autograd
        schema_outputs = nn.utils.rnn.pad_sequence(schema_outputs, batch_first = True)

        # schema linking
        if self.type=='speech_schema':
            # print(outputs.shape)
            # print(schema_outputs.shape)
            sims = self.get_cos_similarity(outputs,schema_outputs)
            outputs = self.get_sw_embedding(outputs, schema_outputs, sims)
        
        
        # outputs = nn.utils.rnn.pad_sequence(schema_outputs, batch_first = True)
        # decoder_speeches_len = [e.shape[0] for e in outputs]
        decoder_speeches_len = [1 for e in outputs]
        # speech_outputs = nn.utils.rnn.pad_sequence(speech_outputs, batch_first = True)

        # encoder_inputs = torch.cat([outputs.unsqueeze(0), schema_outputs], axis = 1)

        # encoder_inputs = torch.cat([outputs, schema_outputs], axis = 1)
        
        
        # attn_mask = None 
        # encoder_inputs_len = None
        # attn_mask = batch.padding_mask(decoder_speeches_len,batch.schema_sents_len)
        # # 啥意思啊？ 为什么要np.sum
        # # encoder_inputs_len = np.sum([batch.decoder_speeches_len, batch.schema_sents_len], axis = 0)
        # encoder_inputs_len = np.sum([decoder_speeches_len, batch.schema_sents_len], axis = 0)
        # pos = True if encoder_inputs_len is not None else False
        # # 有个问题：schema和音频长度不一定对应  老师的做法音频长度是都变成10？
        # encoder_outputs, encoder_attentions = self.encoder(encoder_inputs, inputs_len = encoder_inputs_len, pos = pos, attn_mask = attn_mask)

        # schema_len = schema_outputs.size()[1]
        # schema_outputs = encoder_outputs[:, -schema_len:] 
        # encoder_outputs = encoder_outputs[:, :-schema_len] 
        encoder_outputs = outputs.unsqueeze(1)
        # toc = time.perf_counter()
        # print(f"该程序耗时: {toc - tic:0.4f} seconds")
        return batch, encoder_outputs, schema_outputs


    def forward(self, examples):
        batch, encoder_outputs, schema_outputs = self.get_decoder_inputs(examples)
        decoder_outputs = self.decoder(batch, encoder_outputs, schema_outputs)
        return decoder_outputs, 0
    def parse(self, examples, beam_size):
        batch, encoder_outputs, schema_outputs = self.get_decoder_inputs(examples)
        match_outputs = None
        #match_outputs = self.get_match_outputs(examples, speech_embeddings)
        # tic = time.perf_counter()
        decoder_outputs = self.decoder.parse(batch, encoder_outputs, schema_outputs, beam_size, match_outputs)
        # toc = time.perf_counter()
        # print(f"该程序耗时: {toc - tic:0.4f} seconds")
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

