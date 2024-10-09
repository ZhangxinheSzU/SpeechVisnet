# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : semQL.py
# @Software: PyCharm
"""

Keywords = ['des', 'asc', 'and', 'or', 'sum', 'min', 'max', 'avg', 'none', '=', '!=', '<', '>', '<=', '>=', 'between', 'like', 'not_like'] + [
    'in', 'not_in', 'count', 'intersect', 'union', 'except'] + ['bar', 'pie', 'line', 'scatter', 'stacked', 'grouping'] + ['binning',
        'minute', 'hour', 'day', 'weekday', 'month', 'quarter', 'year', 'UDF'
    ]


class Grammar(object):
    def __init__(self, is_sketch=False):
        self.begin = 0
        self.type_id = 0
        self.is_sketch = is_sketch
        self.prod2id = {}
        self.type2id = {}
        self._init_grammar(Sel)
        self._init_grammar(Root)
        self._init_grammar(Sup)
        self._init_grammar(Filter)
        self._init_grammar(Order)
        self._init_grammar(N)
        self._init_grammar(Root1)
        self._init_grammar(Root0)
        self._init_grammar(Bin)
        self._init_grammar(Vis)
        self._init_grammar(Group)
        #self._init_grammar(V)

        if not self.is_sketch:
            self._init_grammar(A)

        self._init_id2prod()
        self.type2id[C] = self.type_id
        self.type_id += 1
        self.type2id[T] = self.type_id

        #print('type2id', len(self.type2id), self.type2id)
        #print('prod2id', len(self.prod2id), self.prod2id)

    def _init_grammar(self, Cls):
        """
        get the production of class Cls
        :param Cls:
        :return:
        """
        production = Cls._init_grammar()
        for p in production:
            self.prod2id[p] = self.begin
            self.begin += 1
        self.type2id[Cls] = self.type_id
        self.type_id += 1

    def _init_id2prod(self):
        self.id2prod = {}
        for key, value in self.prod2id.items():
            self.id2prod[value] = key

    def get_production(self, Cls):
        return Cls._init_grammar()


class Action(object):
    def __init__(self):
        self.pt = 0
        self.production = None
        self.children = list()
        self.value = None

    def get_next_action(self, is_sketch=False):
        actions = list()
        for x in self.production.split(' ')[1:]:
            if x not in Keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type not in [A]:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def set_parent(self, parent):
        self.parent = parent

    def add_children(self, child):
        self.children.append(child)

class Root0(Action):
    def __init__(self, id_c, parent=None):
        super(Root0, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.repr = 'Root'
    
    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Root0 Root1',
            1: 'Root0 Vis Root1',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Root0(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Root0(' + str(self.id_c) + ')'

 
class Vis(Action):
    def __init__(self, id_c, parent=None):
        super(Vis, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = ' '.join(self.production.split()[1:])
        self.repr = 'Visualize'
    
    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Vis bar',
            1: 'Vis pie',
            2: 'Vis line',
            3: 'Vis scatter',
            4: 'Vis stacked bar',
            5: 'Vis grouping line',
            6: 'Vis grouping scatter'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Vis(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Vis(' + str(self.id_c) + ')'
   


class Root1(Action):
    def __init__(self, id_c, parent=None):
        super(Root1, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = None if self.id_c == 3 else self.production.split()[1]
        self.repr = 'Root1'

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'Root1 intersect Root Root',
            1: 'Root1 union Root Root',
            2: 'Root1 except Root Root',
            3: 'Root1 Root',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Root1(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Root1(' + str(self.id_c) + ')'


class Root(Action):
    def __init__(self, id_c, parent=None):
        super(Root, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.repr = 'Q'

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'Root Sel Sup Filter',
            1: 'Root Sel Filter Order',
            2: 'Root Sel Sup',
            3: 'Root Sel Filter',
            4: 'Root Sel Order',
            5: 'Root Sel',
            6: 'Root Sel Group Sup Filter',
            7: 'Root Sel Group Filter Order',
            8: 'Root Sel Group Sup',
            9: 'Root Sel Group Filter',
            10: 'Root Sel Group Order',
            11: 'Root Sel Group'

        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Root(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Root(' + str(self.id_c) + ')'


class N(Action):
    """
    Number of Columns
    """
    def __init__(self, id_c, parent=None):
        super(N, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'N A',
            1: 'N A A',
            2: 'N A A A',
            3: 'N A A A A',
            4: 'N A A A A A'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'N(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'N(' + str(self.id_c) + ')'

class C(Action):
    """
    Column
    """
    def __init__(self, id_c, parent=None):
        super(C, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self.production = 'C T'
        self.table = None
        self.value = id_c
        self.repr = 'Column'

    def __str__(self):
        return 'C(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'C(' + str(self.id_c) + ')'


class T(Action):
    """
    Table
    """
    def __init__(self, id_c, parent=None):
        super(T, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'T min'
        self.table = None
        self.value = id_c
        self.repr = 'Table'

    def __str__(self):
        return 'T(' + str(self.id_c) + ')'

    def __repr__(self):
        
        return 'T(' + str(self.id_c) + ')'

class V(Action):
    """
    Value
    """
    def __init__(self, value, parent=None):
        super(V, self).__init__()

        self.parent = parent
        self.id_c = value
        self.value = '<value>'  #' '.join(str(value).split())
        self.production = 'V value'
        self.table = None
        self.repr = 'Value'

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'V value',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()    

    def __str__(self):
        #return 'V(' + str(self.id_c) + ')'
        return 'V(0)'


    def __repr__(self):
        
        return 'V(0)'



class A(Action):
    """
    Aggregation
    """
    def __init__(self, id_c, parent=None):
        super(A, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = self.production.split()[1]
        self.repr = 'Aggregation'

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'A none C',
            1: 'A max C',
            2: "A min C",
            3: "A count C",
            4: "A sum C",
            5: "A avg C"
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'A(' + str(self.id_c) + ')'

    def __repr__(self):
        #return 'A(' + str(self.grammar_dict[self.id_c].split(' ')[1]) + ')'
        return 'A(' + str(self.id_c)+ ')'



class Sel(Action):
    """
    Select
    """
    def __init__(self, id_c, parent=None):
        super(Sel, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.repr = 'Select'

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Sel N',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Sel(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Sel(' + str(self.id_c) + ')'

class Filter(Action):
    """
    Filter
    """
    def __init__(self, id_c, parent=None):
        super(Filter, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = self.production.split()[1]
        self.repr = 'Filter'

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            # 0: "Filter 1"
            0: 'Filter and Filter Filter',
            1: 'Filter or Filter Filter',
            2: 'Filter = A',
            3: 'Filter != A',
            4: 'Filter < A',
            5: 'Filter > A',
            6: 'Filter <= A',
            7: 'Filter >= A',
            8: 'Filter between A',
            9: 'Filter like A',
            10: 'Filter not_like A',
            # now begin root
            11: 'Filter = A Root',
            12: 'Filter < A Root',
            13: 'Filter > A Root',
            14: 'Filter != A Root',
            15: 'Filter between A Root',
            16: 'Filter >= A Root',
            17: 'Filter <= A Root',
            # now for In
            18: 'Filter in A Root',
            19: 'Filter not_in A Root'

        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Filter(' + str(self.id_c) + ')'

    def __repr__(self):
        #return 'Filter(' + str(self.grammar_dict[self.id_c]) + ')'
        return 'Filter(' + str(self.id_c) + ')'



class Sup(Action):
    """
    Superlative
    """
    def __init__(self, id_c, parent=None):
        super(Sup, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = self.production.split()[1]
        self.repr = 'Superlative'

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Sup des A',
            1: 'Sup asc A',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Sup(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Sup(' + str(self.id_c) + ')'


class Order(Action):
    """
    Order
    """
    def __init__(self, id_c, parent=None):
        super(Order, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        self.value = self.production.split()[1]
        self.repr = 'Order_by'


    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Order des A',
            1: 'Order asc A',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Order(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Order(' + str(self.id_c) + ')'

class Group(Action):
    """
    group by
    """
    def __init__(self, id_c, parent=None):
        super(Group, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        #self.value = self.production.split()[1]
        self.repr = 'Group_by'


    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Group grouping A',
            1: 'Group binning Bin',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Group(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Group(' + str(self.id_c) + ')'


class Bin(Action):
    """
    bin by
    """
    def __init__(self, id_c, parent=None):
        super(Bin, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]
        if self.id_c <= 6:
            self.value = self.production.split()[-1]
        else:
            self.value = 'UDF'
        self.repr = 'Bin_by'


    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Bin A minute', # bin x by minute
            1: 'Bin A hour',
            2: 'Bin A day',
            3: 'Bin A weekday',
            4: 'Bin A month',
            5: 'Bin A quarter',
            6: 'Bin A year',
            7: 'Bin A N', # bin x into N
            8: 'Bin A UDF', # bin x by values(e.g. 5)
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x


        return self.grammar_dict.values()

    def __str__(self):
        return 'Bin(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Bin(' + str(self.id_c) + ')'



if __name__ == '__main__':
    print(list(Root._init_grammar()))
