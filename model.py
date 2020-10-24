# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:27:48 2019

@author: 86187
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from numpy.random import RandomState


    
    

    
class TeRo(nn.Module):
    def __init__(self, kg, embedding_dim, batch_size, learning_rate, L, gran, gamma, n_day, gpu=True):
        super(TeRo, self).__init__()
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_day = n_day
        self.gran = gran

        self.L = L
        # Nets
        self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_R_real = torch.nn.Embedding(self.kg.n_relation*2, self.embedding_dim, padding_idx=0)
        self.emb_R_img = torch.nn.Embedding(self.kg.n_relation*2, self.embedding_dim, padding_idx=0)
        self.emb_Time = torch.nn.Embedding(n_day, self.embedding_dim, padding_idx=0)
        
        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E_real.weight.data.uniform_(-r, r)
        self.emb_E_img.weight.data.uniform_(-r, r)
        self.emb_R_real.weight.data.uniform_(-r, r)
        self.emb_R_img.weight.data.uniform_(-r, r)
        self.emb_Time.weight.data.uniform_(-r, r)
        # self.emb_T_img.weight.data.uniform_(-r, r)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
        if self.gpu:
            self.cuda()



    def forward(self, X):
        h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.int64)//self.gran

        if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            r_i = Variable(torch.from_numpy(r_i).cuda())
            d_i = Variable(torch.from_numpy(d_i).cuda())
        else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            r_i = Variable(torch.from_numpy(r_i))
            d_i = Variable(torch.from_numpy(d_i))

        pi = 3.14159265358979323846
        d_img = torch.sin(self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        d_real = torch.cos(
            self.emb_Time(d_i).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))

        h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_img

        t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_real-\
                 self.emb_E_img(t_i).view(-1,self.embedding_dim)*d_img


        r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

        h_img = self.emb_E_real(h_i).view(-1, self.embedding_dim) *d_img+\
                 self.emb_E_img(h_i).view(-1,self.embedding_dim) *d_real


        t_img = self.emb_E_real(t_i).view(-1, self.embedding_dim) *d_img+\
                self.emb_E_img(t_i).view(-1,self.embedding_dim) *d_real

        r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)



        if self.L == 'L1':
            out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
            out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
            out = out_real + out_img

        else:
            out_real = torch.sum((h_real + r_real + d_i - t_real) ** 2, 1)
            out_img = torch.sum((h_img + r_img + d_i + t_real) ** 2, 1)
            out = torch.sqrt(out_img + out_real)

        return out

    def normalize_embeddings(self):
        self.emb_E_real.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E_img.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def log_rank_loss(self, y_pos, y_neg, temp=0):
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma-y_pos
        y_neg = self.gamma-y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        if self.gpu:
            loss = loss.cuda()
        return loss


    def rank_loss(self, y_pos, y_neg):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        if self.gpu:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda()
        else:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss



    def rank_left(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    Xe_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2] if triple[3]>=0 else triple[2]+self.kg.n_relation
                        X_i[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                        Xe_i[i, 0] = i
                        Xe_i[i, 1] = triple[1]
                        Xe_i[i, 2] = triple[2]+self.kg.n_relation if triple[4]>=0 else triple[2]
                        Xe_i[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                    i_score = self.forward(X_i)+self.forward(Xe_i)
                    if rev_set>0:
                        X_rev = np.ones([self.kg.n_entity,4])
                        Xe_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2 if triple[3]>=0 else triple[2]+self.kg.n_relation+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                            Xe_rev[i, 0] = triple[1]
                            Xe_rev[i, 1] = i
                            Xe_rev[i, 2] = triple[2]+self.kg.n_relation//2+self.kg.n_relation if triple[4]>=0 else triple[2]+self.kg.n_relation//2
                            Xe_rev[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)+self.forward(Xe_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]                            
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out]=1e6 
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                    rank.append(rank_triple)
                        

            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = self.forward(X_i)
                    if rev_set>0:
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]                            
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out]=1e6 
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                    rank.append(rank_triple)

        return rank

    def rank_right(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    Xe_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2] if triple[3]>=0 else triple[2]+self.kg.n_relation
                        X_i[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                        Xe_i[i, 0] = triple[0] 
                        Xe_i[i, 1] = i
                        Xe_i[i, 2] = triple[2]+self.kg.n_relation if triple[4]>=0 else triple[2]
                        Xe_i[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                    i_score = self.forward(X_i)+self.forward(Xe_i)
                    if rev_set>0: 
                        X_rev = np.ones([self.kg.n_entity,4])
                        Xe_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2 if triple[3]>=0 else triple[2]+self.kg.n_relation+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3] if triple[3]>=0 else triple[4]
                            Xe_rev[i, 0] = i
                            Xe_rev[i, 1] = triple[0]
                            Xe_rev[i, 2] = triple[2]+self.kg.n_relation//2+self.kg.n_relation if triple[4]>=0 else triple[2]+self.kg.n_relation//2
                            Xe_rev[i, 3] = triple[4] if triple[4]>=0 else triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)+ self.forward(Xe_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]       
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out]=1e6
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
        
                    rank.append(rank_triple)
                    
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = self.forward(X_i)
                    if rev_set>0: 
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
        
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]       
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out]=1e6
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
        
                    rank.append(rank_triple)

        return rank

    def timepred(self, X):
        rank = []
        with torch.no_grad():
            for triple in X:
                X_i = np.ones([self.kg.n_day, len(triple)])
                for i in range(self.kg.n_day):
                    X_i[i, 0] = triple[0]
                    X_i[i, 1] = triple[1]
                    X_i[i, 2] = triple[2]
                    X_i[i, 3:] = self.kg.time_dict[i]
                i_score = self.forward(X_i)
                if self.gpu:
                    i_score = i_score.cuda()
    
                target = i_score[triple[3]]           
                rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                rank.append(rank_triple)

        return rank

    


class ATISE(nn.Module):
    def __init__(self, kg, embedding_dim, batch_size, learning_rate, gamma, cmin, cmax, gpu=True):
        super(ATISE, self).__init__()
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.cmin = cmin
        self.cmax = cmax
        # Nets
        self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_E_var = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.emb_R_var = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.emb_TE = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.alpha_E = torch.nn.Embedding(self.kg.n_entity, 1, padding_idx=0)
        self.beta_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.omega_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_TR = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.alpha_R = torch.nn.Embedding(self.kg.n_relation, 1, padding_idx=0)
        self.beta_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.omega_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        
    
        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.uniform_(-r, r)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.uniform_(-r, r)
        self.alpha_E.weight.data.uniform_(0, 0)
        self.beta_E.weight.data.uniform_(0, 0)
        self.omega_E.weight.data.uniform_(-r, r)
        self.emb_TR.weight.data.uniform_(-r, r)
        self.alpha_R.weight.data.uniform_(0, 0)
        self.beta_R.weight.data.uniform_(0, 0)
        self.omega_R.weight.data.uniform_(-r, r)

        # Regularization
        self.normalize_embeddings()
        
        if self.gpu:
            self.cuda()
            
    def forward(self, X):
        h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.float32)

        if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            r_i = Variable(torch.from_numpy(r_i).cuda())
            d_i = Variable(torch.from_numpy(d_i).cuda())

        else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            r_i = Variable(torch.from_numpy(r_i))
            d_i = Variable(torch.from_numpy(d_i))

        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.embedding_dim) \
            + self.beta_E(h_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(h_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))
            
        t_mean = self.emb_E(t_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.embedding_dim) \
            + self.beta_E(t_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(t_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))
            
        r_mean = self.emb_R(r_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.embedding_dim) \
            + self.beta_R(r_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_R(r_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))


        h_var = self.emb_E_var(h_i).view(-1, self.embedding_dim)
        t_var = self.emb_E_var(t_i).view(-1, self.embedding_dim)
        r_var = self.emb_R_var(r_i).view(-1, self.embedding_dim)

        out1 = torch.sum((h_var+t_var)/r_var, 1)+torch.sum(((r_mean-h_mean+t_mean)**2)/r_var, 1)-self.embedding_dim
        out2 = torch.sum(r_var/(h_var+t_var), 1)+torch.sum(((h_mean-t_mean-r_mean)**2)/(h_var+t_var), 1)-self.embedding_dim
        out = (out1+out2)/4
        

        return out
    
    
    
    def log_rank_loss(self, y_pos, y_neg, temp=0):
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma-y_pos
        y_neg = self.gamma-y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        if self.gpu:
            loss = loss.cuda()
        return loss


    def rank_loss(self, y_pos, y_neg):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        if self.gpu:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda()
        else:
            target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)

        
    def regularization_embeddings(self):
        lower = torch.tensor(self.cmin).float()
        upper = torch.tensor(self.cmax).float()
        if self.gpu:
            lower = lower.cuda()
            upper = upper.cuda()
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data<self.cmin,lower,self.emb_E_var.weight.data)
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data>self.cmax,upper,self.emb_E_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data < self.cmin,lower, self.emb_R_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data > self.cmax,upper, self.emb_R_var.weight.data)
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        
    def rank_left(self, X, facts, kg, timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    i_score = torch.zeros(self.kg.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3],triple[4]]:
                        for i in range(0, self.kg.n_entity):
                            X_i[i, 0] = i
                            X_i[i, 1] = triple[1]
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        i_score = i_score + self.forward(X_i).view(-1)
                        if rev_set>0:
                            X_rev = np.ones([self.kg.n_entity,4])
                            for i in range(0, self.kg.n_entity):
                                X_rev[i, 0] = triple[1]
                                X_rev[i, 1] = i
                                X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                                X_rev[i, 3] = time_index
                            i_score = i_score + self.forward(X_rev).view(-1)
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]                            
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out]=1e6 
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                    rank.append(rank_triple)
                        
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = i
                        X_i[i, 1] = triple[1]
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = self.forward(X_i)
                    if rev_set>0:
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = triple[1]
                            X_rev[i, 1] = i
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]
                    target = i_score[int(triple[0])].clone()
                    i_score[filter_out]=1e6
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                    rank.append(rank_triple)
        return rank

    def rank_right(self, X, facts, kg,timedisc, rev_set=0):
        rank = []
        with torch.no_grad():
            if timedisc:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    i_score = torch.zeros(self.kg.n_entity)
                    if self.gpu:
                        i_score = i_score.cuda()
                    for time_index in [triple[3],triple[4]]:
                        for i in range(0, self.kg.n_entity):
                            X_i[i, 0] = triple[0]
                            X_i[i, 1] = i
                            X_i[i, 2] = triple[2]
                            X_i[i, 3] = time_index
                        i_score = i_score + self.forward(X_i).view(-1)
                        if rev_set>0:
                            X_rev = np.ones([self.kg.n_entity,4])
                            for i in range(0, self.kg.n_entity):
                                X_rev[i, 0] = i
                                X_rev[i, 1] = triple[0]
                                X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                                X_rev[i, 3] = time_index
                            i_score = i_score + self.forward(X_rev).view(-1)
                            
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out]=1e6 
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
        
                    rank.append(rank_triple)
            else:
                for triple, fact in zip(X, facts):
                    X_i = np.ones([self.kg.n_entity, 4])
                    for i in range(0, self.kg.n_entity):
                        X_i[i, 0] = triple[0]
                        X_i[i, 1] = i
                        X_i[i, 2] = triple[2]
                        X_i[i, 3] = triple[3]
                    i_score = self.forward(X_i)
                    if rev_set>0: 
                        X_rev = np.ones([self.kg.n_entity,4])
                        for i in range(0, self.kg.n_entity):
                            X_rev[i, 0] = i
                            X_rev[i, 1] = triple[0]
                            X_rev[i, 2] = triple[2]+self.kg.n_relation//2
                            X_rev[i, 3] = triple[3]
                        i_score = i_score + self.forward(X_rev).view(-1)
                    if self.gpu:
                        i_score = i_score.cuda()
                    filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]       
                    target = i_score[int(triple[1])].clone()
                    i_score[filter_out]=1e6
                    rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
        
                    rank.append(rank_triple)

        return rank

    def timepred(self, X):
        rank = []
        with torch.no_grad():
            for triple in X:
                X_i = np.ones([self.kg.n_day, len(triple)])
                for i in range(self.kg.n_day):
                    X_i[i, 0] = triple[0]
                    X_i[i, 1] = triple[1]
                    X_i[i, 2] = triple[2]
                    X_i[i, 3:] = self.kg.time_dict[i]
                i_score = self.forward(X_i)
                if self.gpu:
                    i_score = i_score.cuda()
    
                target = i_score[triple[3]]           
                rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
                rank.append(rank_triple)

        return rank
