#!/usr/bin/env python
# coding=utf-8

import json
import time

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer

from data_process.generate_retrieval.parse_ast import build_tree
from src.rule.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()



def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

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


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):

    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1

def process(data, table):

    process_dict = {}

    origin_sql = data['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

    sql = data.copy()
    sql['pre_sql'] = data

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))

    col_set_type = np.zeros((len(col_set_iter), 4))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names

    return process_dict, sql

def is_valid(rule_label, col_table_dict, sql):
    try:
        build_tree(copy.copy(rule_label))
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
