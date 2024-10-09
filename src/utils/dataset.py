# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""
import math
import copy
import torch
import torch.nn as nn
import numpy as np

import src.rule.semQL as define_rule
from src.models.common_modules import nn_utils
from src.config import cfg

#from bert_serving.client import BertClient
#bc = BertClient(port=5015, port_out=6016)

def speech_padding(speeches):
    '''
        padding to the max length of current batch to ensure the same length
    '''
    new_speeches = []
    max_length = max([len(x) for x in speeches])
    for speech in speeches:
        n_frames = speech.shape[0]
        if max_length > n_frames:
            speech = np.tile(speech,(max_length//n_frames+1, 1))
        speech = speech[:max_length, :]
        new_speeches.append(speech)
    try:
        new_speeches = np.array(new_speeches, dtype = np.float32)
    except Exception as e:
        print(e)
        print('speech len', [len(x) for x in new_speeches])
    new_speeches = torch.from_numpy(new_speeches)
    new_speeches.unsqueeze_(1)
    if cfg.cuda:
        new_speeches = new_speeches.cuda()
    return new_speeches




def gen_x_batch(schema, word_emb, embed_size, cuda = False, word_encoder = None):
    q = schema
    B = len(q)
    val_embs = []
    val_len = np.zeros(B, dtype=np.int64)
    is_list = True
    total_q = []
    unk_words = []
    
    for i, one_q in enumerate(q):
        val_len[i] = len(one_q)

        q_val = []
        lens = []
        for ws in one_q:
            # ws = ws.split()
            emb_list = []
            # ws_len = len(ws)
            for w in ws:
                try:
                    if w not in word_emb:
                        unk_words.append(w)
                    emb_list.append(word_emb.get(w, word_emb['<unk>']))
                except Exception as e:
                    print('---------word {} is not exist in the vocab'.format(w))
                    emb_list.append(word_emb.get(w, np.random.rand(embed_size)))
            
            
            emb_array = np.array(emb_list, dtype = np.int64)
            emb_array = torch.from_numpy(emb_array)
            q_val.append(emb_array)
            lens.append(len(emb_list))

        q_val= nn.utils.rnn.pad_sequence(q_val, batch_first = True)
        if cuda:
            q_val = q_val.cuda()
        _, q_val, _ = word_encoder(q_val, lens)

        val_embs.append(q_val)
    # if unk_words != []:
        # print('unk_words', unk_words)
    return val_embs,val_len.tolist()



class Example:
    """
        class for data unit
    """
    def __init__(self, nl,src_sentence, sql=None,col_set=None, table_names=None, table_len=None, tab_cols = None,
                 col_table_dict=None, cols=None, schema_seqs = None,table_seqs=None,graph = None,tgt_actions=None,tts = None
        ):
        self.nl = nl
        self.src_sentence = src_sentence
        self.src_sentence_len = len(self.src_sentence)
        self.sql = sql
        self.col_set = col_set
        self.schema_seqs = schema_seqs
        self.table_seqs = table_seqs
        # table
        self.table_names = table_names
        self.table_len = table_len
        # column
        self.col_table_dict = col_table_dict
        self.cols = cols # total
        self.tab_cols = tab_cols # set of cols
        self.col_len = len(self.tab_cols)
        self.tts = tts
        # self.graph = graph

        # #print('graph nodes', len(self.graph.nodes.values()), self.graph.nodes.values())
        # true_nodes = [' '.join(x) for x in self.table_names + self.tab_cols]
        # index = true_nodes.index('*')
        # true_nodes[index] = 'all'
        # #print('true nodes', len(true_nodes), true_nodes)

        # if list(self.graph.nodes.values()) != true_nodes:
        #     print('1', list(self.graph.nodes.values()))
        #     print('2', true_nodes)
        # action
        self.tgt_actions = tgt_actions
        self.truth_actions = copy.deepcopy(tgt_actions)
        self.sketch = list()
        if self.truth_actions:
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A) or isinstance(ta, define_rule.V):
                    continue
                self.sketch.append(ta)
        # self.sketch = list()
        # self.sketch_str = []
        # self.label_match = np.zeros(self.table_len + len(self.tab_cols))
        # if self.truth_actions:
        #     # construct sketech for sketch decoder
        #     for ta in self.truth_actions:
        #         if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A):
        #             if isinstance(ta, define_rule.C):
        #                 try:
        #                     index = int(str(ta)[2:-1])
        #                     self.label_match[table_len+index] = 1
        #                 except:
        #                     pass
        #             if isinstance(ta, define_rule.T):
        #                 try:
        #                     index = int(str(ta)[2:-1])
        #                     self.label_match[index] = 1
        #                 except:
        #                     pass
        #             continue
        #         self.sketch.append(ta)
        #         self.sketch_str.append(str(ta))
        # self.sketch_str = ' '.join(self.sketch_str)
        # if self.sketch_str == 'Root1(3) Root(3) Sel(0) N(0) Filter(2)':
        #     self.freq = 1
        # else:
        #     self.freq = 0


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

def get_speech_len_for_decoder(src_speeches_len):
    '''
        get the true length of speech after speech_encoder
    '''
    l = src_speeches_len
    #l = [min(y//32 + 1, 10) for y in l]
    #l = [int((y-(cfg.CNNRNN.kernel_size-cfg.CNNRNN.stride))/cfg.CNNRNN.stride) for y in l]
    #l = [int((y-(cfg.CNNRNN.kernel_size-cfg.CNNRNN.stride))/cfg.CNNRNN.stride) for y in l]

    #l = [min(y, 320) for y in l]
    #kernel_size = 4
    #stride = 2
    #for _ in range(5):
    #    l = [math.ceil((y-(kernel_size - stride)) / stride) for y in l]
    #print(l)
    l = [10 for y in l]
    return l
    
def padding_adjs(nodes, adjs):
    batch_size = nodes.size()[0]
    max_len = nodes.size()[1]
    adjs_padded = np.zeros([batch_size, max_len, max_len], dtype = np.float32)

    for i in range(batch_size):
        adj_len = adjs[i].shape[0]
        adjs_padded[i, :adj_len, :adj_len] = adjs[i]
    adjs = torch.from_numpy(adjs_padded).cuda()
    return adjs

def padding_texts(texts, word2idx = None, go_eos = True):
    if word2idx is not None:
        texts = [[word2idx.get(w, word2idx['<unk>']) for w in seqs] for seqs in texts]
    max_len = max([len(text) for text in texts])
    texts_new = []
    text_lens = []
    for text in texts:
        if go_eos:
            if len(text) == max_len:
                text_lens.append(len(text) + 1)
            else:
                text_lens.append(len(text) + 2)

            text = [word2idx['<go>']] + text + [word2idx['<eos>']] + [word2idx['<pad>']] * (max_len  - len(text))
        else:
            text = text + [word2idx['<pad>']] * (max_len - len(text))

        texts_new.append(text)
    texts_new = np.array(texts_new, dtype = np.int64)
    assert not np.any(np.isnan(texts_new))
    texts_new = torch.from_numpy(texts_new)
    if cfg.cuda:
        texts_new = texts_new.cuda()
    if go_eos:
        return texts_new, text_lens
    else:
        return texts_new

class Batch(object):
    '''
        batch for examples
    '''
    def __init__(self,type, examples, grammar, cuda=False, word_encoder = None):
        self.examples = examples
        #print('len examples', len(self.examples), '----')
        self.type = type
        if examples[0].tgt_actions:
            self.max_action_num = max(len(e.tgt_actions) for e in self.examples) # max SQL rules for one batch 
            self.max_sketch_num = max(len(e.sketch) for e in self.examples) # max SQL sketch rules for one batch
        #print('sketch', [e.sketch for e in self.examples])
        # speech
        # self.src_speeches = nn.utils.rnn.pad_sequence([e.src_speech.squeeze(0) for e in self.examples],batch_first = True)
        # self.src_speeches_len = [e.src_speech_len for e in self.examples]
        # #print(self.src_speeches_len)
        #self.src_speeches_len = [len(e.src_speech) for e in self.examples]
        # self.decoder_speeches_len = get_speech_len_for_decoder(self.src_speeches_len)
        # text
        self.src_sents = [e.src_sentence for e in self.examples]
        self.src_sents_len = [len(e.src_sentence) for e in self.examples]
        # self.text_seqs = [e.src_sent for e in self.examples]
        # if self.type=='text':
            # self.text_seqs_input = padding_texts(self.src_sents, word_emb, go_eos = False)
            # self.text_seqs_embedding, self.text_seqs_len = gen_x_batch(self.src_sents, word_emb,  cfg.SCHEMA.col_embed_size, cuda, word_encoder)
        # schema
        self.schema_sents_word = [e.schema_seqs for e in self.examples]
        # schema_sents_len就是句子长度

        # self.schema_sents_embedding,self.schema_sents_len = gen_x_batch(self.schema_sents_word, word_emb, cfg.SCHEMA.col_embed_size, cuda, word_encoder)
        self.schema_sents_embedding = word_encoder.predict(self.schema_sents_word, source_lang="eng_Latn")
        # self.schema_sents_len = [sents_embedding.shape[0] for sents_embedding in self.schema_sents_embedding]
        self.schema_sents_len = [self.schema_sents_embedding.shape[0]]
        # self.schema_adj = [torch.from_numpy(e.graph.adj) for e in self.examples ]
        #self.schema_adjs = padding_adjs(self.schema_sents_embedding, self.schema_adj)
        # self.schema_sents_len = [len(self.schema_adj[i]) for i in range(len(self.schema_adj))]
        #self.schema_table_len = [e.graph.table_num for e in self.examples]


        self.table_sents = [e.tab_cols for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_len = [e.col_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]



        self.grammar = grammar
        self.cuda = cuda

        #print('nodes len', self.schema_sents_len)
        #print(' column + table len', [x+len(y) for x, y in zip(self.table_len, self.table_sents)])

    def __len__(self):
        return len(self.examples)


    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)
        #return nn_utils.table_dict_to_mask_tensor2(self.schema_sents_len, self.table_len, table_dict, cuda=self.cuda)

    def table_dict_mask2(self, table_dict):
        #return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)
        return nn_utils.table_dict_to_mask_tensor2(self.schema_sents_len, self.table_len, table_dict, cuda=self.cuda)

    def table_dict_mask3(self, table_dict):
        #return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)
        return nn_utils.table_dict_to_mask_tensor3(self.table_len, self.col_len, table_dict, cuda=self.cuda)

    def padding_mask(self,decoder_speeches_len,schema_sents_len):
        speech_lens = decoder_speeches_len
        schema_lens = schema_sents_len
        # 老师这里设置为10
        max_speech_lens = max(speech_lens)
        max_len = max_speech_lens + max(schema_lens)
        batch_size = len(speech_lens)
        pad_mask = np.ones([batch_size, max_len], dtype = np.int64)

        for i in range(batch_size):
            pad_mask[i, :speech_lens[i]] = 0
            pad_mask[i, max_speech_lens: (max_speech_lens + schema_lens[i])] = 0
        if self.cuda:
            pad_mask = torch.from_numpy(pad_mask).bool().cuda()
        else :
            pad_mask = torch.from_numpy(pad_mask).bool()
        pad_mask = pad_mask.unsqueeze(1).expand(-1, max_len, -1)  # shape [B, L_q, L_k]
        if self.cuda:
            pad_mask = pad_mask.cuda()
        return pad_mask


    #@cached_property
    #def pred_col_mask(self):
    #    return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    
    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.schema_sents_len, cuda=self.cuda)

    @cached_property
    def schema_token_mask1(self):
        return nn_utils.length_2array_to_mask_tensor(self.table_len, self.col_len, 
                                                    cuda=self.cuda)
    
    @cached_property
    def schema_token_mask2(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len,
                                                    cuda=self.cuda)

    
    @cached_property
    def col_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_len, cuda=self.cuda)
    
    @cached_property
    def col_token_mask2(self):
        return nn_utils.length_array_to_mask_tensor(self.col_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)



    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_2array_to_mask_tensor(self.decoder_speeches_len, self.schema_sents_len, 
                                                    cuda=self.cuda)

    

    @cached_property
    def src_speech_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.decoder_speeches_len, max_len = 10,
                                                    cuda=self.cuda)


