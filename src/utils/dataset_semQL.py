# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -*- coding: utf-8 -*-
#@summerzhao: 2021/07/30
import json
import copy
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from functools import reduce

from src.config import cfg
from src.rule import semQL as define_rule
from src.models.common_modules import nn_utils
#from src.utils.vocab import Vocab


def gen_sent_embed(sents, word2idx, word_encoder):
    embed_size = 300
    #word_emb = vocab.embedding
    sents_emb = []
    lens = []
    unk_words = []
    for sent in sents:
        assert sent != ''
        emb_list = []
        sent_len = len(sent)
        for word in sent:
            try:
                if word not in word2idx:
                    unk_words.append(word)
                #emb = word_emb[word2idx.get(word, word2idx['<unk>'])]
                emb = word2idx.get(word, word2idx['<unk>'])
                emb_list.append(emb)
            except Exception as e:
                print('---------word {} is not exist in the vocab {}'.format(word, sent))
                emb_list.append(np.random.rand(embed_size))
        #emb_array = np.array(emb_list, dtype = np.float32)
        emb_array = np.array(emb_list, dtype = np.int64)
        assert not np.any(np.isnan(emb_array))
        emb_array = torch.from_numpy(emb_array)

        sents_emb.append(emb_array)
        lens.append(len(emb_list))
    #sents_emb = torch.stack(sents_emb, axis = 0)
    sents_emb = nn.utils.rnn.pad_sequence(sents_emb, batch_first = True)
    sents_emb = sents_emb
    if cfg.cuda:
        sents_emb = sents_emb.cuda()
    #sents_emb = word_encoder(sents_emb)
    if word_encoder is not None:
        outputs, sents_emb = word_encoder(sents_emb, lens)
    
    if unk_words != []:
        print('unk_words', unk_words)
    
    lens = np.array(lens, dtype = np.int64)
    return outputs, sents_emb, lens


def gen_x_batch(sents, word2idx, word_encoder = None, return_embed = False):
    # get embedding for each sentence in sents
    if return_embed:
        #sents = [list(reduce(lambda x, y: x + y, sent)) for sent in sents]
        sents_emb, _, sents_len = gen_sent_embed(sents, word2idx, word_encoder)
        return sents_emb, sents_len
    
    # ge embedding for each word in each sentence in sents
    batch_size = len(sents)
    sents_emb = []
    sents_len = np.zeros(batch_size, dtype=np.int64)
    for i, sent in enumerate(sents):
        sents_len[i] = len(sent)
        _, sent_emb, _ = gen_sent_embed(sent, word2idx, word_encoder)
        sents_emb.append(sent_emb)
    
    sents_emb = nn.utils.rnn.pad_sequence(sents_emb, batch_first = True)
    if cfg.cuda:
        sents_emb = sents_emb.cuda()

    return sents_emb, sents_len



class Example(object):
    """
        class for data unit
    """
    def __init__(self, 
                 src_sent, tgt_actions = None, vis_seq = None, tab_cols = None, col_num = None, sql = None,
                 one_hot_type = None, col_hot_type = None, schema_len = None, tab_ids = None,
                 table_names = None, table_len = None, col_table_dict = None, cols = None,
                 table_col_name = None, table_col_len = None, col_pred = None, tokenized_src_sent = None,
                 record_name = None, text = None, sql_seqs = None, node_seqs = None, adj = None, text_seqs = None,
                 schema_seqs = None, node_seqs_neg = None, adj_neg = None, sql_neg = None, rouge = None):
        self.record_name = record_name
        self.text = text
        self.sql = sql
        self.sql_seqs = sql_seqs
        self.node_seqs = node_seqs
        self.adj = adj
        self.text_seqs = text_seqs
        self.schema_seqs = schema_seqs
        self.node_seqs_neg = node_seqs_neg
        self.adj_neg = adj_neg
        self.sql_neg = sql_neg
        self.rouge = rouge

        self.src_sent = src_sent
        self.tokenized_src_sent = tokenized_src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.one_hot_type=one_hot_type
        self.col_hot_type = col_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.col_pred = col_pred
        self.tgt_actions = tgt_actions
        self.truth_actions = copy.deepcopy(tgt_actions)

        self.sketch = list()
        if self.truth_actions:
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A) or isinstance(ta, define_rule.V):
                    continue
                self.sketch.append(ta)
         
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

def padding_adjs(nodes, adjs):
    batch_size = nodes.size()[0]
    max_len = nodes.size()[1]
    adjs_padded = np.zeros([batch_size, max_len, max_len], dtype = np.float32)

    for i in range(batch_size):
        adj_len = adjs[i].shape[0]
        adjs_padded[i, :adj_len, :adj_len] = adjs[i]
    adjs = torch.from_numpy(adjs_padded)
    if cfg.cuda:
        adjs = adjs.cuda()
    return adjs

def get_src_map(texts, vocab = None):
    if isinstance(texts[0][0], list):
        texts = [[x[0] for x in text] for text in texts]
    texts_index = padding_texts(texts, vocab.word2idx, go_eos = False)
    batch_size = texts_index.size()[0]
    texts_len = texts_index.size()[1]
    vocab_size = vocab.size
    src_map = torch.zeros([batch_size, texts_len, vocab_size])
    if cfg.cuda:
        src_map = src_map.cuda()
    for i, text_index in enumerate(texts_index):
        for j, index in enumerate(text_index):
            src_map[i, j, index] = 1
    return src_map

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



class Batch(object):
    '''
        batch for examples
    '''
    def __init__(self, examples, vocab, word_encoder = None, return_embed = None, test = False, revision = False):
        self.examples = examples
        self.texts = [e.text for e in self.examples]
        self.sqls = [e.sql for e in self.examples]
        self.record_names = [e.record_name for e in self.examples]

        # input for text encoder
        #if cfg.useVis:
        if True:
            self.text_seqs = [e.src_sent for e in self.examples]
            self.text_seqs_input = padding_texts([[x[0] for x in text_seq] for text_seq in self.text_seqs], vocab.word2idx, go_eos = False)
            self.text_seqs_embedding, self.text_seqs_len = gen_x_batch(self.text_seqs, vocab.word2idx, word_encoder, return_embed = False)
        else:
            self.text_seqs = [e.text_seqs for e in self.examples]
            self.text_seqs_input = padding_texts([[x[0] for x in text_seq] for text_seq in self.text_seqs], vocab.word2idx, go_eos = False)
            self.text_seqs_embedding, self.text_seqs_len = gen_x_batch(self.text_seqs, vocab.word2idx, word_encoder, return_embed = True)
        
        self.schema_seqs = [e.schema_seqs for e in self.examples]
        self.schema_seqs_embedding, self.schema_seqs_len = gen_x_batch(self.schema_seqs, vocab.word2idx, word_encoder)


        # input for ast encoder
        self.node_seqs_pos = [e.node_seqs for e in self.examples]
        self.node_seqs_embedding_pos, self.node_seqs_len_pos = gen_x_batch(self.node_seqs_pos, vocab.word2idx, word_encoder, return_embed = True)
        self.adjs_pos = [np.array(e.adj, dtype = np.int64) for e in self.examples]
        #self.adjs_pos = padding_adjs(self.node_seqs_embedding_pos, self.adjs_pos)
        


        self.node_seqs_neg = [e.node_seqs_neg for e in self.examples]
        self.node_seqs_embedding_neg, self.node_seqs_len_neg = gen_x_batch(self.node_seqs_neg, vocab.word2idx, word_encoder, return_embed = True)
        self.adjs_neg = [np.array(e.adj_neg, dtype = np.int64) for e in self.examples]
        #self.adjs_neg = padding_adjs(self.node_seqs_embedding_neg, self.adjs_neg)
        #print('node_seqs', self.node_seqs_neg[0],self.node_seqs_embedding_neg.size())

        if (not revision) and test:
            # shuffle the order
            self.tags = []
            node_seqs_embedding1, node_seqs_embedding2 = [], []
            adjs1, adjs2 = [], []
            node_seqs_len1, node_seqs_len2 = [], []
            for i in range(len(self.examples)):
                node_seqs_len_pos = self.node_seqs_len_pos[i]
                node_seqs_len_neg = self.node_seqs_len_neg[i]
                if random.random() > 0.5:
                    self.tags.append(0) # positive, negtive
                    node_seqs_embedding1.append(self.node_seqs_embedding_pos[i][:node_seqs_len_pos])
                    adjs1.append(self.adjs_pos[i])
                    node_seqs_len1.append(node_seqs_len_pos)
                    node_seqs_embedding2.append(self.node_seqs_embedding_neg[i][:node_seqs_len_neg])
                    adjs2.append(self.adjs_neg[i])
                    node_seqs_len2.append(node_seqs_len_neg)
                else:
                    self.tags.append(1) # negtive, postive
                    node_seqs_embedding1.append(self.node_seqs_embedding_neg[i][:node_seqs_len_neg])
                    adjs1.append(self.adjs_neg[i])
                    node_seqs_len1.append(node_seqs_len_neg)
                    node_seqs_embedding2.append(self.node_seqs_embedding_pos[i][:node_seqs_len_pos])
                    adjs2.append(self.adjs_pos[i])
                    node_seqs_len2.append(node_seqs_len_pos)

            node_seqs_embedding1 = nn.utils.rnn.pad_sequence(node_seqs_embedding1, batch_first = True)
            node_seqs_embedding2 = nn.utils.rnn.pad_sequence(node_seqs_embedding2, batch_first = True)

            self.node_seqs_embeddings = [node_seqs_embedding1, node_seqs_embedding2]

            self.adjs1 = padding_adjs(node_seqs_embedding1, adjs1)
            self.adjs2 = padding_adjs(node_seqs_embedding2, adjs2)
            self.adjs = [self.adjs1, self.adjs2]
            node_seqs_len1 = np.array(node_seqs_len1, dtype = np.int64)
            node_seqs_len2 = np.array(node_seqs_len2, dtype = np.int64)
            self.node_seqs_lens = [node_seqs_len1, node_seqs_len2]
            
        else:
            # keep positive ahead
            self.node_seqs_embeddings = [self.node_seqs_embedding_pos, self.node_seqs_embedding_neg]

            self.adjs_pos = padding_adjs(self.node_seqs_embedding_pos, self.adjs_pos)
            self.adjs_neg = padding_adjs(self.node_seqs_embedding_neg, self.adjs_neg)

            self.adjs = [self.adjs_pos, self.adjs_neg]
            self.node_seqs_lens = [self.node_seqs_len_pos, self.node_seqs_len_neg]

            
            self.tags = [0] * len(examples) 


        #self.src_map = get_src_map(self.node_seqs, vocab)


        # target for revision
        if revision:
            self.node_seqs_embedding_pattern = self.node_seqs_embedding_neg
            self.adjs_pattern = self.adjs_neg
            self.node_seqs_len_pattern = self.node_seqs_len_neg
            self.rouges = [e.rouge for e in examples]
            self.rouges = torch.from_numpy(np.array(self.rouges, dtype = np.float32))
            if cfg.cuda:
                self.rouges = self.rouges.cuda()
            self.sql_seqs = [e.sql_seqs for e in examples]
            self.targets, self.sql_seqs_len = padding_texts(self.sql_seqs, vocab.word2idx_out, go_eos = True)
            self.targets_mask = self.get_target_mask(self.schema_seqs, vocab.word2idx_out, self.targets, vocab.idx2word_out)
            #self.targets_mask = None


        if examples[0].tgt_actions:
            self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
            self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]

        #self.grammar = grammar
        self.cuda = cfg.cuda
     


    def __len__(self):
        return len(self.examples)

    def save_batch(self):
        data = pd.DataFrame({'text': self.texts, 'sql': self.sqls, 'node_seq': self.node_seqs, 'sbt_seq': self.sbt_seqs, 'text_seq': self.text_seqs})
        data['node_seq'] = data['node_seq'].apply(lambda x: ' '.join(['_'.join(s) for s in x]))
        data['sbt_seq'] = data['sbt_seq'].apply(lambda x: ' '.join(['_'.join(s) for s in x]))

        data['text_seq'] = data['text_seq'].apply(lambda x: ' '.join(x))
        data.to_json('./batch.txt', orient = 'records', indent = 1)

    def padding_mask(self):
        text_lens = self.text_seqs_len
        sql_lens = self.node_seqs_len_pattern
        max_len = max(text_lens) + max(sql_lens)
        batch_size = len(text_lens)
        pad_mask = np.ones([batch_size, max_len], dtype = np.int64)

        for i in range(batch_size):
            pad_mask[i, :text_lens[i]] = 0
            pad_mask[i, max(text_lens): (max(text_lens) + sql_lens[i])] = 0
        pad_mask = torch.from_numpy(pad_mask).bool()
        pad_mask = pad_mask.unsqueeze(1).expand(-1, max_len, -1)  # shape [B, L_q, L_k]
        if cfg.cuda:
            pad_mask = pad_mask.cuda()
        return pad_mask

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    def get_target_mask(self, schema_seqs, word2idx, targets, idx2word):
        batch_size = len(schema_seqs)
        mask = np.ones((batch_size, len(word2idx)), dtype = np.int64)
        keywords = cfg.VOCAB.structure_tokens
        for batch, schema_seq in enumerate(schema_seqs):
            schema_words = reduce(lambda x, y: x +y, schema_seq)
            target_words = list(set(schema_words + keywords))
            target_words = [word2idx.get(word, word2idx['<unk>']) for word in target_words]
            
            #exception_words = [word for word in targets[batch] if word not in target_words]
            #if len(exception_words) >= 1:
            #    print(exception_words, schema_words, [idx2word[word] for word in exception_words])
            for index in target_words:
                mask[batch][index] = 0
        mask = torch.from_numpy(mask).bool().unsqueeze(1)
        if cfg.cuda:
            mask = mask.cuda()
        return mask

    @cached_property
    def pred_col_mask(self):
        return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        if not cfg.useVis:
            return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                        cuda=self.cuda)
        else:
            return nn_utils.length_2array_to_mask_tensor(self.src_sents_len, self.node_seqs_len_pattern, cuda = self.cuda)


        


