#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/13
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common_modules import Attention
from src.config import cfg

def gen_A(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    if cfg.cuda:
        adj = adj.cuda()
    return adj

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    input: 
        inputs
        adj: A + I_n
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        '''
            inputs: len=batch, [node1*embedding, node2*embedding, ...]
            adj: len=batch, [node1 * node1, node2*node2] different example with different adj
            output: batch * node * out_features
        '''
        #support = torch.matmul(input, self.weight) # batch * node * out_features
        outputs = []
        lens = []
        for i in range(len(inputs)):
            #print(inputs[i].size(), self.weight.size())
            support = torch.matmul(inputs[i], self.weight)
            adj_nor = gen_adj(adj[i]) # node * node
            output = torch.matmul(adj_nor, support) # node * out_features
            if self.bias is not None:
                output =  output + self.bias
            output = F.relu(output)
            outputs.append(output)
            lens.append(len(output))
        #outputs = nn.utils.rnn.pad_sequence(outputs, batch_first = True)
        #outputs = nn.utils.rnn.pack_sequence(outputs, enforce_sorted=False) 
        return outputs, lens

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MultiLayerGCN(nn.Module):
    def __init__(self, num_layers):
        super(MultiLayerGCN, self).__init__()
        self.gcns = nn.ModuleList(
          [GCN(cfg.SCHEMA_GCN.in_features, cfg.SCHEMA_GCN.out_features, True) for _ in range(num_layers)])


    def forward(self, inputs, adj):
        for gcn in self.gcns:
            inputs, lens = gcn(inputs, adj)
        outputs = nn.utils.rnn.pad_sequence(inputs, batch_first = True)
        return outputs, lens





class SchemaEncoder(nn.Module):
    def __init__(self):
        super(SchemaEncoder, self).__init__()
        if cfg.SCHEMA.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(cfg.SCHEMA_RNN.input_size, cfg.SCHEMA_RNN.hidden_size, cfg.SCHEMA_RNN.num_layers, batch_first = True,
                               dropout = cfg.SCHEMA_RNN.dropout, bidirectional = cfg.SCHEMA_RNN.bidirectional)
        elif cfg.CNNRNN.rnn_type == 'GRU':
            self.rnn = nn.GRU(cfg.SCHEMA_RNN.input_size, cfg.SCHEMA_RNN.hidden_size, cfg.SCHEMA_RNN.num_layers, batch_first = True, 
                              dropout = cfg.SCHEMA_RNN.dropout, bidirectional = cfg.SCHEMA_RNN.bidirectional)
        else:
            print('schema rnn_type is not legal')
            raise NotImplementedError
        self.gcn = GCN(cfg.SCHEMA_GCN.in_features, cfg.SCHEMA_GCN.out_features)
        self.att = Attention.MultiHeadAttention(model_dim = cfg.SCHEMA_ATT.hidden_size, num_heads = cfg.SCHEMA_ATT.n_heads)
        self.fc = nn.Linear(cfg.SCHEMA_GCN.out_features, cfg.SCHEMA_ATT.hidden_size)

    def forward(self, schema_inputs, schema_adjs):
        #print('SchemaEncoder inputs:', len(schema_inputs), schema_inputs[0].size())
        x, lens = self.gcn(schema_inputs, schema_adjs)
        #if cfg.SCHEMA_RNN.use:
        #    x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True, enforce_sorted=False)
        #    x, hx = self.rnn(x)
        #    x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        #else:
        #    print('------------------- not use rnn')
        x = self.fc(x)
        x = F.relu(x)

        #if cfg.SCHEMA_ATT.self_att:
        #    attn_mask = Attention.padding_mask(x, x, cfg.cuda)
        #    x, attn_weights = self.att(x, x, x, attn_mask)
        #print('SchemaEncoder outputs: ', x.size())
        return x


