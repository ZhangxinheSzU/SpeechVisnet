import librosa
import json
import math
import numpy as np
import os
import copy
import scipy.signal
import pickle
import random
from nltk.stem import WordNetLemmatizer
import logging
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from src.utils import dataset_semQL, utils_semQL
#from src.utils.audio import load_audio, get_audio_length, audio_with_sox, augment_audio_with_sox, load_randomly_augmented_audio
from src.utils import utils_audio
from src.utils.dataset_test import Example, Batch
from src.rule import lf
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1, Root0, Bin, V, Vis, Group
from preprocess.process_spider.schema2graph import TableGraph
from src.config import cfg


wordnet_lemmatizer = WordNetLemmatizer()
windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def process(sql, speech, table):

    process_dict = {}
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]
    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]
    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    #col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    # graph = TableGraph(table['nodes'], table['nodes_type'], np.array(table['adj'], dtype = np.float32))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    #process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names
    process_dict['speech'] = speech
    # process_dict['graph'] = graph
    return process_dict

def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def is_valid(rule_label, col_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
            except:
                flag = True
                print(sql['question'])
    return flag is False

def to_example(sql, speech, table, ifeval = False):
    process_dict = process(sql, speech, table)

    col_table_dict = get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
    #table_col_name = get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

    #process_dict['col_set_iter'][0] = ['count', 'number', 'many']

    rule_label = None
    if 'rule_label' in sql:
        try:
            rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
        except Exception as e:
            print(e)
            return None
        if is_valid(rule_label, col_table_dict = col_table_dict, sql=sql) is False:
            return None
    schema = sql['names'][1:] + sql['table_names']
    schema_seqs = [[wordnet_lemmatizer.lemmatize(v.lower()) for v in x.split()] for x in schema]
    example = Example(
            src_speech = process_dict['speech'],
            #col_num=len(process_dict['col_set_iter']),
            tab_cols=process_dict['col_set_iter'],
            sql=sql['query'],
            table_names=process_dict['table_names'],
            table_len=len(process_dict['table_names']),
            col_table_dict=col_table_dict,
            cols=process_dict['tab_cols'],
            schema_seqs = schema_seqs,
            tgt_actions=rule_label,
            # graph = process_dict['graph']
        )
    example.sql_json = copy.deepcopy(sql)
    if cfg.mode == 'train_augment' and not ifeval:
        '''
            sample the most frequent type
        '''
        if example.freq == 1:
            sample_rate = np.random.rand()
            if sample_rate > 1:
                return None
    return example


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(object):
    # normalize 改成True
    def __init__(self, normalize=True, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.normalize = normalize
        self.audio_feature = cfg.audio_feature
        self.hop_length = cfg.DATA_SPEECH.hop_length
        self.n_fft = cfg.DATA_SPEECH.n_fft
        self.sample_rate = cfg.DATA_SPEECH.sample_rate
        self.window = windows.get(cfg.DATA_SPEECH.window, windows['hamming'])
        if os.path.exists(cfg.scalar_path):
            self.scalar = pickle.load(open(cfg.scalar_path, 'rb'))
        else:
            self.scalar = None


    def parse_audio(self, audio_path, ifeval):
        if not os.path.exists(audio_path):
            print('path not exist', audio_path)
            return  None
        try:
            y, sr = librosa.load(audio_path, sr=22050)
        except:
            return  None

        if cfg.add_noise:
            noise = np.random.randn(len(y))
            y = y + 0.01 * noise

        
        n_fft = self.n_fft
        win_length = n_fft
        hop_length = self.hop_length


        if self.audio_feature == 'magphase':
            features = utils_audio.get_audio_magphase(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window, normalize = self.normalize)
        elif self.audio_feature in ['fbank', 'cmvn']:
            features = utils_audio.get_audio_fbank(y, sr = sr, n_mels = cfg.SPEECH.input_dim, n_fft = n_fft, hop_length = hop_length)
        elif self.audio_feature == 'mfcc':
            features = utils_audio.get_audio_mfcc(y, sr = sr, n_mfcc = cfg.SPEECH.input_dim, n_fft = n_fft, hop_length = hop_length)
        elif self.audio_feature == 'all':
            features = utils_audio.get_audio_all(y, sr = sr, n_fft = n_fft, hop_length = hop_length, window = self.window, normalize = self.normalize)
        else:
            raise ValueError("audio feature, need to be one of 'magphase/fbank/cmvn/mfcc', but it's {}".format(cfg.audio_feature))
        #print(features, features.shape)
        spect = features
        if spect is None:
            return spect
        # 为什么大于320就要去掉？先注释掉康康吧
        # if not ifeval and spect.shape[1] > 320:
        #     return None
        
        if self.normalize:
            spect = spect.transpose(1, 0)
            spect = self.scalar.transform(spect)
            spect = spect.transpose(1, 0)
            spect = self.pad_spect(spect)

        return spect


    def parse_spect(self, audio_path, ifeval):
        if not os.path.exists(audio_path):
            print('path not exist', audio_path)
            return  None
        try:
            with open(audio_path, 'rb') as f:
                spect = pickle.load(f)
        except Exception as e:
            print(audio_path)
            print(e)
            return  None

        if spect is None:
            return spect
        # 为什么大于320就要去掉？先注释掉康康吧
        # if not ifeval and spect.shape[1] > 320:
        #     return None
        spect = spect.transpose(1, 0)
        # spect = self.scalar.transform(spect)
        # spect = spect.transpose(1, 0)
        # spect = self.pad_spect(spect)
        #print('spect', spect.shape)
        return spect

    def pad_spect(self, spect):
        spect_pad = np.zeros([96, 320])
        size = min(spect.shape[1], 320)
        spect_pad[:, :size] = spect[:, :size]
        return spect_pad

class SpeechSQLDataset(Dataset, SpectrogramParser):
    def __init__(self, sql_path, speech_path, table_path, normalize=False, augment=False, ifeval = False, parse_audio = False, mode = 'train'):
        '''
            self-defined sub class for speech &sql data
        '''
        super(SpeechSQLDataset, self).__init__()
        self.ifeval = ifeval
        self.mode = mode
        # load table data
        with open(table_path) as f:
            print("Loading data from %s"%table_path)
            table_data = json.load(f)
        self.table_data = {table['db_id']: table for table in table_data}

        # load sql data
        self.sql_data = []
        print("Loading data from %s" % sql_path)
        with open(sql_path) as inf:
            data = json.load(inf)
            data = lower_keys(data)
            self.sql_data += data#[:10000]
            print('total data:', len(self.sql_data))


        self.speech_path = speech_path
        self.dev_speech_path = cfg.dev_wav_path
        self.train_speech_path = '/home/zhoujh/speech2SQL/tts/wav_train_spect/{}.pkl'

        #if len(self.sql_data) > 9000:
        #    self.sql_data = self.sql_data[9000:19000]
        self.ifeval = ifeval
        self.if_parse_audio = parse_audio
        if ifeval:
            self.sql_data = self.sql_data#[2000:]     

 
    def __getitem__(self, index):
        index = index % self.__len__()
        sql_data = self.sql_data[index]
        table_data = self.table_data[sql_data['db_id']]
        tts_filename = sql_data['filename']

        if self.mode == 'dev':
            wav_path = self.dev_speech_path.format(tts_filename)
        else:
            wav_path = self.train_speech_path.format(tts_filename)
        
        # for add noise
        #if self.mode == 'dev':
        #    wav_path = '/data/app/summerzhao/speech2SQL/data/speechSQL/dev/noise/noise_90dB_spect/{}.pkl'.format(tts_filename)

        # for another tts
        # if self.mode == 'dev':
        #     wav_path = '/data/app/summerzhao/speech2SQL/tts_baidu/wav_dev_spect/{}.pkl'.format(tts_filename)

        if self.if_parse_audio:
            spect = self.parse_audio(wav_path, self.ifeval)
        else:
            spect = self.parse_spect(wav_path, self.ifeval)
            
        #print(spect.shape)

        if spect is None:
            return None
        example = to_example(sql_data, spect, table_data, self.ifeval)
        return example


    def __len__(self):
        return len(self.sql_data)


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(
            noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


def _collate_fn(batch):

    # filter example which is None
    batch = list(filter(lambda x : x is not None, batch))
    #print('len batch', len(batch))

    # descending sorted
    batch = sorted(batch, key=lambda sample: len(sample.src_speech), reverse=True)
    return batch


class SpeechSQLDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechSQLDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
