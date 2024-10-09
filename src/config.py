#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/13
import os
import argparse
import os.path as osp
import numpy as np
from easydict import EasyDict as edict


parser = argparse.ArgumentParser(description='speech2sql training')
parser.add_argument('--log_name', default='speech2sql.log', type=str, help="training log file")
parser.add_argument('--model_save_path', default='speech2sql', type=str, help="training model file")
parser.add_argument('--load_model', default='/home/zhoujh/speech2SQL/saved_model/augment_split/speech2sql/epoch100.model', type=str, help="continuous training model file")
parser.add_argument('--eval_load_model', default='', type=str, help="eval model")
parser.add_argument('--load_pretrained_model', default='', type=str, help="pretraining model file")
parser.add_argument('--GPU', default='5', type=str, help="GPU device used")
parser.add_argument('--num_workers', default=1, type=int, help="num workers used")
parser.add_argument('--toy', action='store_true', help="if use toy data")
parser.add_argument('--parse_speech', action='store_true', help="if parse speech during training")
parser.add_argument('--pretrain', action='store_true', help="if load pretrained speech part")
parser.add_argument('--schema_rnn_use', action='store_true', help="if use rnn module in schema encoder")
parser.add_argument('--epoch', default=300, type=int, help="num epoches used")
parser.add_argument('--batch_size', default=2, type=int, help="num batch_size used")
parser.add_argument('--lr', default=1e-6, type=float, help="learning rate")
parser.add_argument('--speech_type', default='', type=str, help="model type for speech encoder")
parser.add_argument('--parallel', action='store_true', help="if parallel to train the model")
parser.add_argument('--schema_linking', action='store_true', help="if use schema linking with speech")
parser.add_argument('--device_ids', default=None, nargs='+', type=int, help="model type for speech encoder")
parser.add_argument('--eval_save_path', default='./results/predict_lf.json',  type=str, help="model results save")
parser.add_argument('--mode', default='train',  type=str, help="mode of dataset, train or train_augment")
parser.add_argument('--audio_feature', default='fbank',  type=str, help="audio feature type, mfcc/fbank/cmvn/magphase")
parser.add_argument('--word_encoder', default=None,  type=str, help="lstm /glove /bert")
parser.add_argument('--use_gcn', action='store_true', help="if use gcn for schema encoder")
parser.add_argument('--use_match', action='store_true', help="if use gcn for schema encoder")
parser.add_argument('--no_transformer', action='store_true', help="if use gcn for schema encoder")
parser.add_argument('--sample_type', default='', help="type of sampling data")
parser.add_argument('--data_dir', default='./data/nvbench/', type=str, help="dataset dir")
parser.add_argument('--transformer_layers', default=2,  type=int, help="lstm /glove /bert")
parser.add_argument('--transformer_heads', default=4,  type=int, help="lstm /glove /bert")
parser.add_argument('--gcn_layers', default=2,  type=int, help="lstm /glove /bert")
parser.add_argument("--wav_path",default = '',type=str,help='')
parser.add_argument("--db_id",default='',type = str, help='')
parser.add_argument('--LSTM_num_layers', default=1, type=int, help="LSTM_num_layers")
parser.add_argument('--heads', default=4, type=int, help="heads_num")






args = parser.parse_args()

__C = edict()
cfg = __C

# public
__C.cuda = True
__C.toy = args.toy
__C.parse_speech = args.parse_speech
__C.pretrain = args.pretrain
__C.num_workers = args.num_workers
__C.schema_linking = args.schema_linking
__C.audio_feature = args.audio_feature
__C.add_noise = False
__C.word_encoder = args.word_encoder
__C.use_gcn = args.use_gcn
__C.ifMatch = args.use_match
__C.use_transformer = not args.no_transformer

# dataset
__C.mode = args.mode
__C.dataset = 'nvbench'
__C.dataset_dir = args.data_dir
sample_type = args.sample_type


__C.table_path = './data/nvbench/tables.json'
#__C.table_path = os.path.join(cfg.dataset_dir, "tables_new_nocolumn.json")
__C.train_sql_path = os.path.join(cfg.dataset_dir, "%s_SemQL.json"%__C.mode if sample_type == '' else '{}_{}_SemQL.json'.format(__C.mode, sample_type))
__C.train_mfcc_path = os.path.join(cfg.dataset_dir, "train_mfcc.pkl")
# if __C.audio_feature == 'fbank' and __C.dataset == 'spider':
#     __C.train_wav_path = './tts_en/spider/train_augment/wav_spect/{}.pkl'
#     __C.dev_wav_path = './tts_en/spider/dev/wav_spect/{}.pkl'
# else:
# __C.train_wav_path = './tts_en/%s/%s/wav/{}.wav'%(__C.dataset, __C.mode)
# __C.dev_wav_path = './tts_en/%s/%s/wav/{}.wav'%(__C.dataset, __C.mode)
__C.train_wav_path = './data/%s/%s/wav/{}.wav'%(__C.dataset, __C.mode)
__C.dev_wav_path = './data/%s/%s/wav/{}.wav'%(__C.dataset, 'dev')
__C.test_speech_path = './data/%s/%s/wav/{}.wav'%(__C.dataset, 'test')
__C.dev_sql_path = os.path.join(cfg.dataset_dir, "dev_SemQL.json" if sample_type == '' else 'dev_{}_SemQL.json'.format(sample_type) )
# __C.test_sql_path = os.path.join(cfg.dataset_dir, 'test_{}_SemQL.json'.format(sample_type) )
__C.test_sql_path = os.path.join(cfg.dataset_dir, 'test_SemQL.json' )
__C.dev_sql_path_new = os.path.join(cfg.dataset_dir, './test_new/dev_new.json')
__C.test_sql_path_new = os.path.join(cfg.dataset_dir, './test_new/test_new.json')


__C.dev_mfcc_path = os.path.join(cfg.dataset_dir, "dev_mfcc.pkl")
#__C.dev_wav_path = './tts_en/spider/dev/wav/{}.wav'

__C.glove_embed_path = os.path.join(cfg.dataset_dir, "../glove/glove.42B.300d.txt")
__C.glove_embed_size = 300
__C.bert_embed_size = 1024

__C.train_speech_path = __C.train_wav_path #if __C.parse_speech else __C.train_mfcc_path
__C.dev_speech_path = __C.dev_wav_path #if __C.parse_speech else __C.dev_mfcc_path

__C.test_wav_path = './data/nvbench/test/wav/wav/{}.wav'
# __C.dev_wav_path = './data/speechSQL/dev/wav_spect/{}.pkl'
#__C.log_name = 'pretrain_schema_nornn.log'




# feature extraction
__C.DATA_SPEECH = edict()
__C.DATA_SPEECH.sample_rate = 22000
__C.DATA_SPEECH.hop_length = 512 #帧移
__C.DATA_SPEECH.n_fft = 1024 #帧长
__C.DATA_SPEECH.window = 'hamming'
__C.DATA_SPEECH.normalize = True

# model parameter
## speech
__C.SPEECH = edict()
__C.SPEECH.speech_type = args.speech_type
#__C.SPEECH.style= 'mel'   #npy | WAV
#__C.SPEECH.model = 'CRNN'   #CNN | CRNN | RNN
dims_dict = {'mfcc': 39, 'magphase': 161, 'fbank': 96, 'cmvn': 40, 'all': 240}
__C.SPEECH.max_len = 512
__C.SPEECH.self_att = True
__C.SPEECH.window_size = 25
__C.SPEECH.stride = 10
__C.SPEECH.input_dim = dims_dict[__C.audio_feature]
__C.SPEECH.hidden_size = 512
__C.SPEECH.embedding_dim = 1024
__C.SPEECH.num_layers = 2
__C.SPEECH.sample = 16000

__C.CNNRNN = edict()
__C.CNNRNN.rnn_type = 'GRU'
__C.CNNRNN.in_channels = __C.SPEECH.input_dim
__C.CNNRNN.hid_channels = 50
__C.CNNRNN.hid2_channels = 64
__C.CNNRNN.out_channels = 128  #64
__C.CNNRNN.kernel_size = 4
__C.CNNRNN.stride = 2
__C.CNNRNN.padding = 0

__C.CNNRNN_RNN = edict()
__C.CNNRNN_RNN.input_size = 128     #64
__C.CNNRNN_RNN.hidden_size = 256
__C.CNNRNN_RNN.num_layers = 1
__C.CNNRNN_RNN.dropout = 0.0
__C.CNNRNN_RNN.bidirectional = True

__C.CNNRNN_ATT = edict()
# mychange *2 ->*3
__C.CNNRNN_ATT.hidden_size = __C.CNNRNN_RNN.hidden_size * 4 if __C.CNNRNN_RNN.bidirectional else __C.CNNRNN_RNN.hidden_size 
__C.CNNRNN_ATT.n_heads = 1


# transforer for speech
__C.SPEECH_TRANSFORMER = edict()
__C.SPEECH_TRANSFORMER.num_layers = 4
__C.SPEECH_TRANSFORMER.num_heads = 8
__C.SPEECH_TRANSFORMER.dim_model = 512
__C.SPEECH_TRANSFORMER.dim_key = 64
__C.SPEECH_TRANSFORMER.dim_value = 64
__C.SPEECH_TRANSFORMER.dim_input = 5120
__C.SPEECH_TRANSFORMER.dim_inner = 2048
__C.SPEECH_TRANSFORMER.dropout = 0.1
__C.SPEECH_TRANSFORMER.src_max_len = 2000


## schema
__C.SCHEMA = edict()
__C.SCHEMA.rnn_type = 'GRU'
__C.SCHEMA.col_embed_size = __C.bert_embed_size if __C.word_encoder == 'bert' else 512
__C.SCHEMA.action_embed_size = 128


__C.SCHEMA_GCN = edict()
__C.SCHEMA_GCN.in_features = __C.SCHEMA.col_embed_size
__C.SCHEMA_GCN.out_features = __C.SCHEMA_GCN.in_features
__C.SCHEMA_GCN.num_layers = args.gcn_layers

__C.SCHEMA_RNN = edict()
__C.SCHEMA_RNN.use = args.schema_rnn_use
#print('shcema rnn use', __C.SCHEMA_RNN.use)
__C.SCHEMA_RNN.input_size = __C.SCHEMA_GCN.out_features
__C.SCHEMA_RNN.hidden_size = 256
__C.SCHEMA_RNN.num_layers = 2
__C.SCHEMA_RNN.dropout = 0
__C.SCHEMA_RNN.bidirectional = True



__C.SCHEMA_ATT = edict()
#__C.SCHEMA_ATT.in_size = 1024
# mychange *2 ->*3
__C.SCHEMA_ATT.hidden_size = __C.SCHEMA_RNN.hidden_size * 4 if __C.SCHEMA_RNN.bidirectional else __C.SCHEMA_RNN.hidden_size
#__C.SCHEMA_ATT.hidden_size = __C.SCHEMA_ATT.hidden_size if __C.SCHEMA_RNN.use else __C.SCHEMA_GCN.out_features
__C.SCHEMA_ATT.n_heads = 1
__C.SCHEMA_ATT.self_att = True



# transformer
__C.TRANSFORMER = edict()
__C.TRANSFORMER.hidden_size = __C.CNNRNN_ATT.hidden_size 
__C.TRANSFORMER.n_heads = args.transformer_heads
__C.TRANSFORMER.dim_feedforward = 1024
__C.TRANSFORMER.dropout = 0.3
__C.TRANSFORMER.max_seq_len = 2000 # for position embedding
__C.TRANSFORMER.num_layers = args.transformer_layers




## decoder
__C.DECODER = edict()
__C.DECODER.column_pointer = False
__C.DECODER.action_embed_size = __C.SCHEMA.action_embed_size # size of action embeddings
__C.DECODER.att_vec_size = __C.SCHEMA_ATT.hidden_size # size of attentional vector
__C.DECODER.type_embed_size = 128 # size of word embeddings
__C.DECODER.hidden_size = __C.DECODER.att_vec_size 
__C.DECODER.readout = 'identity' # 'non_linear'
__C.DECODER.col_embed_size = __C.SCHEMA.col_embed_size
__C.DECODER.dropout = 0.3
__C.DECODER.column_att = 'dot_prod' #'affine'
__C.DECODER.decode_max_time_step = 50

# optimizer
__C.optimizer = 'Adam'
__C.epoch = args.epoch
__C.batch_size = args.batch_size
__C.beam_size = 10
__C.lr = args.lr
__C.lr_scheduler = False
__C.lr_scheduler_gammar = 0.5
__C.clip_grad = 5
__C.loss_epoch_threshold = 20
__C.sketch_loss_coefficient = 0.2
__C.load_model = args.load_model
 # pretrained model
__C.load_pretrained_model =  args.load_pretrained_model #'./pretrain/end2end-asr-pytorch-master/saved_model/libri_speech/best_model.th'
__C.eval_load_model = args.eval_load_model #'./saved_model/pretrain1620362561/best_model.model'
__C.model_save_path = args.model_save_path #'./pretrain'
__C.scalar_path = '/data/app/summerzhao/speech2SQL/pretrain/auto-encoder/saved_models/scalar_augment.pkl'


# pretrained
__C.SPEECH_PRETRAIN = edict()
__C.SPEECH_PRETRAIN.hidden_size = 128
__C.SPEECH_PRETRAIN.sample_prob = 0


# text autoencoder
__C.TAE = edict()
__C.TAE.dim_emb = 300 # input_size
__C.TAE.droupout = 0.2
__C.TAE.dim_h = 300 # hidden_size
__C.TAE.num_layers = 2
__C.TAE.dim_z = 1024 # output size
__C.TAE.dropout = 0.2
__C.glove_path = './data/glove.42B.300d.txt'




