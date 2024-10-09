#!/usr/bin/env python
# coding=utf-8
#@summerzhap:2021/7/30
import re
import json
import copy
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# import torchvision.transforms as transforms

from src.config import args, cfg
from src.utils import dataset_semQL, utils_semQL
from data_process.generate_retrieval.parse_ast import eval_rule
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1, Root0, Bin, V, Vis, Group

wordnet_lemmatizer = WordNetLemmatizer()

def remove(text):
    remove_chars1 = '["\',()]+'
    text = re.sub(remove_chars1, '', text)
    remove_chars2 = '[.-_]+'
    text = re.sub(remove_chars2, ' ', text)
    return text

def deal_vql(vql):
    vql = vql.lower()
    vql = vql.replace('constructorstandings', 'constructor standings')
    remove_chars1 = '[,]+'
    vql = re.sub(remove_chars1, ' ', vql)
    try:
        while '.' in vql:
            vql = ' '.join(vql.split('.'))
        while '(' in vql:
            vql = ' '.join(vql.split('('))
        while ')' in vql:
            vql = ' '.join(vql.split(')'))
        while '_' in vql:
            vql = ' '.join(vql.split('_'))
    except Exception as e:
        print(e, vql)
        return None
    vql_seqs = vql.split()
    vql_seqs = list(filter(len, vql_seqs))
    return vql_seqs

def to_example(data, table):
    # postive
    node_seqs = list(data['node_seqs'].values())
    # my change 默认是第一句哈
    text = data['nl_queries'][0]
    if node_seqs == [] or text == '' or text is None:
        print('data is wrong', data['record_name'])
        return None

    # schema-linking prepartion
    process_dict, data = utils_semQL.process(data, table)
    for c_id, col_ in enumerate(process_dict['col_set_iter']):
        for q_id, ori in enumerate(process_dict['q_iter_small']):
            if ori in col_:
                process_dict['col_set_type'][c_id][0] += 1

    utils_semQL.schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                    process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], data)

    col_table_dict = utils_semQL.get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], data)
    table_col_name = utils_semQL.get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])
    process_dict['col_set_iter'][0] = ['count', 'number', 'many']
    
    if 'rule_label' in data:
        try:
            rule_label = eval_rule(data['rule_label'])
        except Exception as e:
            print('no rule label', e, data['rule_label'])
            return None
        if utils_semQL.is_valid(rule_label, col_table_dict=col_table_dict, sql=data) is False:
            # print(rule_label)
            return None
    
    # deal data to sequence
    node_seqs = [wordnet_lemmatizer.lemmatize(x.lower()) for x in node_seqs]

    text = remove(text.lower())
    text_seqs = [wordnet_lemmatizer.lemmatize(x) for x in text.split()]

    if 'query_novalue' in data:
        sql_seqs = deal_vql(data['query_novalue'])
    else:
        sql_seqs = deal_vql(data['query'])
    sql_seqs = [wordnet_lemmatizer.lemmatize(x) if x != 'as' else 'as' for x in sql_seqs]

    schema = data['names'][1:] + data['table_names']
    schema_seqs = [[wordnet_lemmatizer.lemmatize(v.lower()) for v in x.split()] for x in schema]

    # negtive
    node_seqs_neg = list(data['VQL_neg'][0]['node_seqs'].values())
    node_seqs_neg = [wordnet_lemmatizer.lemmatize(x.lower()) for x in node_seqs_neg]
    adj_neg = data['VQL_neg'][0]['adj']

    try:
        rouge = data['VQL_neg'][0]['rouge']
    except:
        rouge = None
    example = dataset_semQL.Example(
        src_sent = process_dict['question_arg'],
        col_num = len(process_dict['col_set_iter']),
        vis_seq = (data['nl_queries'], process_dict['col_set_iter'], data['query']),
        tab_cols = process_dict['col_set_iter'],
        one_hot_type = process_dict['one_hot_type'],
        col_hot_type = process_dict['col_set_type'],
        table_names = process_dict['table_names'],
        table_len = len(process_dict['table_names']),
        col_table_dict = col_table_dict,
        cols = process_dict['tab_cols'],
        table_col_name = table_col_name,
        table_col_len = len(table_col_name),
        tokenized_src_sent = process_dict['col_set_type'],
        tgt_actions = rule_label,
        record_name = data['record_name'],
        text = data['nl_queries'],
        sql = data['query'],
        sql_seqs = sql_seqs,
        node_seqs = node_seqs,
        adj = data['adj'],
        text_seqs = text_seqs, 
        schema_seqs = schema_seqs,
        node_seqs_neg = node_seqs_neg,
        adj_neg = adj_neg,
        sql_neg = data['VQL_neg'][0]['VQL'],
        rouge = rouge
    )
    example.sql_json = copy.deepcopy(data)
    return example

def get_lens(data):
    nl_lens = [len(x['nl_queries'].split()) if x['nl_queries'] is not None else 0 for x in data]
    max_len = np.mean(nl_lens)
    print('--------- mean text len', max_len, self.__len__())

    nl_lens = [len(x['VQL'].split()) if x['VQL'] is not None else 0 for x in data]
    max_len = np.mean(nl_lens)
    print('--------- mean sql len', max_len, self.__len__())


def get_tables(table_path):
    with open(table_path, 'r') as f:
        tables = json.load(f)
    data_table = {table['db_id']: table for table in tables}
    return data_table


class VisTextDataset(Dataset):
    def __init__(self, data_path, table_path = None, retrieval = False):
        super(VisTextDataset, self).__init__()
        self.data = pd.read_json(data_path, orient = 'records')#[:1000]
        if 'query_novalue' in self.data.columns:
            print('-------------- loading query novalue for predict -------------')
        self.data_table = get_tables(table_path)
        top = cfg.top if retrieval else 1
        if retrieval:
            self.data = self.reconstruct(self.data, top)
        self.data.dropna(subset = ['nl_queries'], inplace = True)
        self.data.reset_index(inplace = True)
        print('loading top {} data from {}, length {}'.format(top, data_path, len(self.data)))
        #get_lens(self.data)

    def __getitem__(self, index):
        data = self.data.loc[index]
        db_id = data['db_id']
        table = self.data_table[db_id]
        example = to_example(data, table)
        return example

    def __len__(self):
        return len(self.data)
    
    def reconstruct(self, data, top = 5):
        new_data = []
        for i in range(len(data)):
            case = data.loc[i].to_dict()
            VQL_neg = case['VQL_neg']
            for j in range(min(len(VQL_neg), top)):
                case['VQL_neg'] = [VQL_neg[j]]
                new_data.append(case.copy())
        new_data = pd.DataFrame.from_dict(new_data)
        return new_data



def _collate_fn(batch):

    # filter example which is None
    batch = list(filter(lambda x : x is not None, batch))

    # descending sorted
    #batch = sorted(batch, key=lambda sample: len(sample.text_seqs), reverse=True)
    return batch


class VisTextDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(VisTextDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn



