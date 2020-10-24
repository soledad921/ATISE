# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:11:52 2019

@author: 86187
"""

import model as KGE
from Dataset import KnowledgeGraph
from Dataset_YG import KnowledgeGraphYG


import torch
import numpy as np
from time import time
from sklearn.utils import shuffle as skshuffle
import os

def mean_rank(rank):
    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r


def mrr(rank):
    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr


def hit_N(rank, N):
    hit = 0
    for i in rank:
        if i <= N:
            hit = hit + 1

    hit = hit / len(rank)

    return hit

def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    X_shuff = X.copy()
    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


def sample_negatives(X, C, kg):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
        
    """
    M = X.shape[0]
    X_corr = X
    for i in range(C-1):
        X_corr = np.concatenate((X_corr,X),0)
    X_corr[:int(M*C/2),0]=torch.randint(kg.n_entity,[int(M*C/2)])        
    X_corr[int(M*C/2):,1]=torch.randint(kg.n_entity,[int(M*C/2)]) 

    return X_corr


def sample_negatives_t(X, C, n_day):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 4, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 4, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]
    X_corr = X
    for i in range(C-1):
        X_corr = torch.cat((X_corr,X),0)
    X_corr[:,3]=torch.randint(n_day,[int(M*C)])        


    return X_corr



def train(task ='LinkPrediction',
          modelname='ATISE',
          data_dir='yago',
          dim=500,
          batch=512,
          lr=0.1,
          max_epoch=5000,
          min_epoch=250,
          gamma=1,
          L = 'L1',
          negsample_num=10,
          timedisc = 0,
          lossname = 'logloss',
          cmin = 0.001,
          cuda_able = True,
          rev_set = 1,
          temp = 0.5,
          gran = 7,
          count = 300
          ):

    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)


    """
    Data Loading
    """
    if data_dir == 'yago' or data_dir == 'wikidata':
        kg = KnowledgeGraphYG(data_dir=data_dir, count = count,rev_set = rev_set)
        n_day = kg.n_time
        min_epoch=50
    elif data_dir=='icews14':
        n_day = 365
        kg = KnowledgeGraph(data_dir=data_dir,gran=gran,rev_set = rev_set)
    elif data_dir == 'icews05-15':
        n_day = 4017
        kg = KnowledgeGraph(data_dir=data_dir,gran=gran,rev_set = rev_set)      


    """
    Create a model
    """

    if modelname== 'TERO':
        model = KGE.TeRo(kg, embedding_dim=dim, batch_size=batch, learning_rate=lr, gamma=gamma, L=L, gran=gran, n_day=kg.n_time,gpu=cuda_able)
    if modelname=='ATISE':
        model = KGE.ATISE(kg, embedding_dim=dim, batch_size=batch, learning_rate=lr, gamma=gamma, cmin=cmin, cmax=100*cmin, gpu=cuda_able)

    if modelname == 'ATISE':
        solver = torch.optim.Adam(model.parameters(), model.learning_rate)
        optimizer = 'Adam'
    else:
        solver = torch.optim.Adagrad(model.parameters(), model.learning_rate)
        optimizer = 'Adagrad'
    

    if timedisc == 0 or timedisc ==2:
        train_pos = np.array(kg.training_triples)
        validation_pos = np.array(kg.validation_triples)
        test_pos = np.array(kg.test_triples)
        
    elif timedisc == 1:
        train_pos = []
        validation_pos = []
        test_pos = []
        for fact in kg.training_triples:
            for time_index in range(fact[3],fact[4]+1):
                train_pos.append([fact[0], fact[1], fact[2], time_index])
        train_pos = np.array(train_pos)
       # for fact in kg.validation_triples:
       #     for time_index in range(fact[3],fact[4]+1):
       #         validation_pos.append([fact[0], fact[1], fact[2], time_index])
        validation_pos = np.array(kg.validation_triples)
       # for fact in kg.test_triples:
       #     for time_index in range(fact[3],fact[4]+1):
       #         test_pos.append([fact[0], fact[1], fact[2], time_index])
       # test_pos = np.array(test_pos)        
        test_pos = np.array(kg.test_triples)

        
    losses = []
    mrr_std = 0
    C = negsample_num
    patience = 0
    path = os.path.join(data_dir,modelname,'timediscrete{:.0f}/dim{:.0f}/lr{:.4f}/neg_num{:.0f}/{:.0f}day/gamma{:.0f}/cmin{:.4f}'
                        .format(timedisc,dim,lr,negsample_num,gran,gamma,cmin))
    if timedisc: path = os.path.join(path,'{:.0f}count'.format(count))
    try: 
        os.makedirs(path)
    except:
        print('path existed')
        return
    
    
        
    """
    Training Process
    """
    for epoch in range(max_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('————————————————')
        it = 0
        train_triple = list(get_minibatches(train_pos, batch, shuffle=True))
        for iter_triple in train_triple:
            if iter_triple.shape[0] < batch:
                break
            start = time()
            if task=='TimePrediction':
                iter_neg = sample_negatives_t(iter_triple, C, n_day)
            else:
                iter_neg = sample_negatives(iter_triple, C, kg)
            if timedisc == 2:
                end_miss = np.where(iter_triple[:,4:5]<0)[0]
                start_miss = np.where(iter_triple[:,3:4]<0)[0]
                neg_end_miss = np.where(iter_neg[:,4:5]<0)[0]
                neg_start_miss = np.where(iter_neg[:,3:4]<0)[0]
                
                
                iter_triple_e = np.delete(iter_triple,3,1)
                iter_triple = np.delete(iter_triple,4,1)
                
                iter_triple_e[:,2:3] += kg.n_relation

                iter_triple_e[end_miss,:]=iter_triple[end_miss,:]
                iter_triple[start_miss,:]=iter_triple_e[start_miss,:]
                
                
                iter_neg_e = np.delete(iter_neg,3,1)
                iter_neg = np.delete(iter_neg,4,1)
                
                iter_neg_e[:,2:3] += kg.n_relation
                
                iter_neg_e[neg_end_miss,:]=iter_neg[neg_end_miss,:]
                iter_neg[neg_start_miss,:]=iter_neg_e[neg_start_miss,:]
                

            pos_score = model.forward(iter_triple)
            neg_score = model.forward(iter_neg)
            if timedisc ==2:
                pos_score += model.forward(iter_triple_e)
                neg_score += model.forward(iter_neg_e)
                
            if lossname == 'logloss':
                loss = model.log_rank_loss(pos_score, neg_score,temp=temp)
            else:
                loss = model.rank_loss(pos_score, neg_score)
            losses.append(loss.item())

            solver.zero_grad()
            loss.backward()
            solver.step()


            if lossname == 'marginloss':
                model.normalize_embeddings()
            if modelname == 'ATISE':
                model.regularization_embeddings()

            end = time()

            if it % 33 == 0:
                print('Iter-{}; loss: {:.4f};time per batch:{:.4f}s'.format(it, loss.item(), end - start))

            it += 1

        """
        Evaluation for Link Prediction
        """

        if ((epoch+1)//min_epoch>epoch//min_epoch and epoch < max_epoch) :
            if task == 'LinkPrediction':
                rank = model.rank_left(validation_pos,kg.validation_facts,kg,timedisc,rev_set=rev_set)
                rank_right = model.rank_right(validation_pos,kg.validation_facts,kg,timedisc,rev_set=rev_set)
                rank = rank + rank_right
            else:
                rank = model.timepred(validation_pos)

            m_rank = mean_rank(rank)
            mean_rr = mrr(rank)
            hit_1 = hit_N(rank, 1)
            hit_3 = hit_N(rank, 3)
            hit_5 = hit_N(rank, 5)
            hit_10 = hit_N(rank, 10)
            print('validation results:')
            print('Mean Rank: {:.0f}'.format(m_rank))
            print('Mean RR: {:.4f}'.format(mean_rr))
            print('Hit@1: {:.4f}'.format(hit_1))
            print('Hit@3: {:.4f}'.format(hit_3))
            print('Hit@5: {:.4f}'.format(hit_5))
            print('Hit@10: {:.4f}'.format(hit_10))
            f = open(os.path.join(path, 'result{:.0f}.txt'.format(epoch)), 'w')
            f.write('Mean Rank: {:.0f}\n'.format(m_rank))
            f.write('Mean RR: {:.4f}\n'.format(mean_rr))
            f.write('Hit@1: {:.4f}\n'.format(hit_1))
            f.write('Hit@3: {:.4f}\n'.format(hit_3))
            f.write('Hit@5: {:.4f}\n'.format(hit_5))
            f.write('Hit@10: {:.4f}\n'.format(hit_10))
            for loss in losses:
                f.write(str(loss))
                f.write('\n')
            f.close()
            if mean_rr < mrr_std and patience<3:
                patience+=1
            elif (mean_rr < mrr_std and patience>=3) or epoch==max_epoch-1:
                if epoch == max_epoch-1:
                    torch.save(model.state_dict(), os.path.join(path, 'params.pkl'))
                model.load_state_dict(torch.load(os.path.join(path,'params.pkl')))
                if task == 'LinkPrediction':
                    rank = model.rank_left(test_pos,kg.test_facts,kg,timedisc,rev_set=rev_set)
                    rank_right = model.rank_right(test_pos,kg.test_facts,kg,timedisc,rev_set=rev_set)
                    rank = rank + rank_right
                else:
                    rank = model.timepred(test_pos)


                m_rank = mean_rank(rank)
                mean_rr = mrr(rank)
                hit_1 = hit_N(rank, 1)
                hit_3 = hit_N(rank, 3)
                hit_5 = hit_N(rank, 5)
                hit_10 = hit_N(rank, 10)
                print('test result:')
                print('Mean Rank: {:.0f}'.format(m_rank))
                print('Mean RR: {:.4f}'.format(mean_rr))
                print('Hit@1: {:.4f}'.format(hit_1))
                print('Hit@3: {:.4f}'.format(hit_3))
                print('Hit@5: {:.4f}'.format(hit_5))
                print('Hit@10: {:.4f}'.format(hit_10))
                if epoch == max_epoch-1:
                    f = open(os.path.join(path, 'test_result{:.0f}.txt'.format(epoch)), 'w')
                else:
                    f = open(os.path.join(path, 'test_result{:.0f}.txt'.format(epoch)), 'w')
                f.write('Mean Rank: {:.0f}\n'.format(m_rank))
                f.write('Mean RR: {:.4f}\n'.format(mean_rr))
                f.write('Hit@1: {:.4f}\n'.format(hit_1))
                f.write('Hit@3: {:.4f}\n'.format(hit_3))
                f.write('Hit@5: {:.4f}\n'.format(hit_5))
                f.write('Hit@10: {:.4f}\n'.format(hit_10))
                for loss in losses:
                    f.write(str(loss))
                    f.write('\n')
                f.close()
                break
            if mean_rr>=mrr_std:
                
                torch.save(model.state_dict(), os.path.join(path, 'params.pkl'))
                mrr_std = mean_rr
                patience = 0


