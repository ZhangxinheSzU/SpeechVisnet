import os
import traceback
import re
import sys
import json
import sqlite3
import sqlparse
import random
from os import listdir, makedirs
from collections import OrderedDict
from nltk import word_tokenize, tokenize
from os.path import isfile, isdir, join, split, exists, splitext

from process_sql import get_sql

class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, table):
        #self._schema = schema
        self._table = table
        self._idMap = self._map(self._table)
        self._columnMap = self._table_column_map(self._table)

    #@property
    #def schema(self):
    #    return self._schema

    @property
    def idMap(self):
        return self._idMap
    
    @property
    def columnMap(self):
        return self._columnMap


    def _map(self, table):
        #column_names_original = table['column_names_original']
        #table_names_original = table['table_names_original']
        #
        #for i, (tab_id, col) in enumerate(column_names_original):
        #    if tab_id == -1:
        #        idMap = {'*': i}
        #    else:
        #        key = table_names_original[tab_id].lower()
        #        val = col.lower()
        #        idMap[key + "." + val] = i

        #for i, tab in enumerate(table_names_original):
        #    key = tab.lower()
        #    idMap[key] = i

        idMap = {'*': 0}
        i = 1
        for key, vals in table.items():
            for val in vals:
                key = key.lower()
                val = val.lower()
                idMap[key + "." + val] = i
                i = i+1
        
        for key in table:
            key = key.lower()
            idMap[key] = i
            i = i+1
        return idMap

    def _table_column_map(self, table):
        idMap = {}
        i = 1
        for table_index, key in enumerate(table.keys()):
            idMap[table_index] = []
            for val in table[key]:
                idMap[table_index].append(i)
                i = i + 1

        return idMap


    


def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        #print(db)
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema
    #print(schemas)
    return schemas, db_names, tables



def build_col_key_map(entry):
    column_names_original = entry["column_names_original"]
    table_names_original = entry["table_names_original"]

    # rebuild cols corresponding to idmap in Schema
    def get_column_map():
        idMap = {} 
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap[i] = '__all__'
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[i] = "__" + key.lower() + "." + val.lower() + "__"
        return idMap

    def get_table_map():
        idMap = {}
        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[i] = "__" + key.lower() + "__"

        return idMap

    column_map = get_column_map()
    table_map = get_table_map()

    #print('column_map', column_map)
    #print('table_map', table_map)
    return column_map, table_map


def build_foreign_col_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_col_key_map(entry)
    return tables





if __name__ == '__main__':
    
    sql = "SELECT name ,  country ,  age FROM singer ORDER BY age DESC"
    db_id = "concert_singer"
    table_file = "../../data/spider/tables.json"
    
    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    print(schema)
    sql_label = get_sql(schema, sql)
    print(sql_label)



