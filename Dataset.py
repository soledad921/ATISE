# encoding: utf-8
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict



class KnowledgeGraph:
    def __init__(self, data_dir, gran=1,rev_set=0):
        self.data_dir = data_dir
        self.entity_dict = {}
        self.gran = gran
        self.entities = []
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.training_facts = []
        self.validation_facts = []
        self.test_facts = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0
        self.rev_set = rev_set
        self.start_date = '2014-01-01' if self.data_dir == 'icews14' else '2005-01-01'
        self.start_sec = time.mktime(time.strptime(self.start_date,'%Y-%m-%d'))
        self.n_time=365 if self.data_dir == 'icews14' else 4017
        self.to_skip_final = {'lhs': {}, 'rhs': {}}
        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        self.load_filters()
        '''construct pools after loading'''
        # self.training_triple_pool = set(self.training_triples)
        # self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        if self.rev_set>0: self.n_relation *= 2
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
            day = int((end_sec - self.start_sec) / (self.gran*24 * 60 * 60))
            self.training_triples.append([self.entity_dict[triple[0]],self.entity_dict[triple[2]],self.relation_dict[triple[1]],day])
            self.training_facts.append([self.entity_dict[triple[0]],self.entity_dict[triple[2]],self.relation_dict[triple[1]],triple[3],0])
            if self.rev_set>0: self.training_triples.append([self.entity_dict[triple[2]],self.entity_dict[triple[0]],self.relation_dict[triple[1]]+self.n_relation//2,day])

        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
            day = int((end_sec - self.start_sec) / (self.gran*24 * 60 * 60))
            self.validation_triples.append([self.entity_dict[triple[0]],self.entity_dict[triple[2]],self.relation_dict[triple[1]],day])
            self.validation_facts.append([self.entity_dict[triple[0]],self.entity_dict[triple[2]],self.relation_dict[triple[1]],triple[3],0])

        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
            day = int((end_sec - self.start_sec) / (self.gran*24 * 60 * 60))
            self.test_triples.append(
                    [self.entity_dict[triple[0]], self.entity_dict[triple[2]], self.relation_dict[triple[1]], day])
            self.test_facts.append([self.entity_dict[triple[0]],self.entity_dict[triple[2]],self.relation_dict[triple[1]],triple[3],0])

        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))


    def load_filters(self):
        print("creating filtering lists")
        to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
        facts_pool = [self.training_facts,self.validation_facts,self.test_facts]
        for facts in facts_pool:
            for fact in facts:
                to_skip['lhs'][(fact[1], fact[2],fact[3], fact[4])].add(fact[0])  # left prediction
                to_skip['rhs'][(fact[0], fact[2],fact[3], fact[4])].add(fact[1])  # right prediction
                
        for kk, skip in to_skip.items():
            for k, v in skip.items():
                self.to_skip_final[kk][k] = sorted(list(v))
        print("data preprocess completed")
        
        
