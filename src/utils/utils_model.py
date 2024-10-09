# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch


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
    return ret


def eval_acc(preds, sqls):
    sketch_correct, best_correct = 0, 0
    for i, (pred, sql) in enumerate(zip(preds, sqls)):
        if pred['model_result'] == sql['rule_label']:
            best_correct += 1
    print(best_correct / len(preds))
    return best_correct / len(preds)


def save_checkpoint(model, checkpoint_name):
    torch.save(model.state_dict(), checkpoint_name)


def save_args(args, path):
    with open(path, 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

def init_log_checkpoint_path(args):
    save_path = args.model_save_path
    dir_name = save_path
    save_path = os.path.join(os.path.curdir, 'saved_model/augment_split/', dir_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path

def combine_sem_sql(lf_file, sql_file, save_file):
    data_lf = pd.read_json(lf_file, orient = 'records')
    sqls = []
    with open(sql_file, 'r') as f:
        for line in f.readlines():
            sqls.append(line.strip('\n'))
    data_lf['query_predict'] = sqls
    data_lf[ ['db_id', 'question', 'query', 'query_predict', 'rule_label', 'sketch_result', 'model_results', 'sketch_true', 'lf_true']].to_json(save_file, orient = 'records', indent = 1)

def combine_sem_sql1(lf_file, sql_file, save_file):
    data_lf = pd.read_json(lf_file, orient = 'records')
    sqls = []
    
    data_sql = pd.read_json(sql_file, orient = 'records')
    #sqls = data_sql['predictSQL'].values.tolist()
    #scores = data_sql['exact'].values.tolist()
    data_lf['query_predict'] = data_sql['predictSQL']
    data_lf['exact'] = data_sql['exact']
    data_lf['hardness'] = data_sql['hardness']
    data_lf[ ['db_id', 'question', 'query', 'query_predict', 'rule_label', 'sketch_result', 'model_result', 'sketch_true', 'lf_true', 'exact', 'hardness']].to_json(save_file, orient = 'records', indent = 1)



def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

if __name__ == '__main__':
    sql_data, speech_data, table_data, _, _, _ = load_dataset(True)
    perm=np.random.permutation(len(sql_data))
    st, ed = 0, 4
    examples = to_batch_seq(sql_data, speech_data, table_data, perm, st, ed, is_train=True)

