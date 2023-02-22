import logging
import os
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import utils
from utils import save_model, load_model
from encoder import BERTEncoder, BiEncoder, CrossEncoder

from classifier import BinaryClassifier, ClassAttentionBinaryClassifier, InstanceAttentionBinaryClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import random

class BEBC(nn.Module):
    def __init__(self, args):
        super(BEBC, self).__init__()
        '''
        BEBC: BiEncoder BinaryClassifier
        '''
        self.args = args
        
        self.bert = BERTEncoder(args.model_name_or_path, sent_embed_type=args.pooling)

        self.encoder = BiEncoder(encoder=self.bert, max_seq_length=args.max_seq_length)
        
        self.layer_norm = nn.LayerNorm(self.encoder.hidden_size)
        
        self.classifier = BinaryClassifier(args, feat_dim = self.encoder.hidden_size)
            
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        
    def forward(self, x_support, x_query):

        interplay_matrix = self.encoder(x_query=x_query, x_support=x_support)
        
        interplay_matrix = self.layer_norm(interplay_matrix)

        scores = self.classifier(interplay_matrix) 

        return scores
    
    def predict(self, x_support, x_query, y_support):
        
        # x_support[0] = 'list the cities from which northwest flies'
        
        num_sup_samples = len(x_support)
        
        num_q_samples = len(x_query)
        
        scores  = self.forward(x_support, x_query)
        
        scores = scores.view(num_q_samples, num_sup_samples)
        
        max_score_idx = torch.argmax(scores, dim=1)
       
        max_score_idx = max_score_idx.tolist()
        
        y_q_hat = [y_support[idx] for idx in max_score_idx]
        
        return y_q_hat
    
class CEBC(nn.Module):
    def __init__(self, args):
        super(CEBC, self).__init__()
        self.bert = BERTEncoder(args.model_name_or_path, sent_embed_type=args.pooling)
        self.pair_encoder = CrossEncoder(encoder=self.bert, max_seq_length=args.max_seq_length)
        self.classifier = BinaryClassifier(args=None, feat_dim=self.pair_encoder.hidden_size)
    
    def forward(self, x_query, x_support):
        z = self.pair_encoder(x_query=x_query, x_support=x_support)
        y_hat = self.classifier(z)
        return y_hat
    
    def predict(self, x_query, x_support, y_support):
        n_support = len(x_support)
        n_query = len(x_query)
        scores = self.forward(x_query=x_query, x_support=x_support)
        scores = scores.view(n_query, n_support)
        max_score_idx = torch.argmax(scores, dim=1)
        max_score_idx = max_score_idx.tolist()
        y_q_hat = [y_support[idx] for idx in max_score_idx]
        return y_q_hat
    
class Random(nn.Module):
    def __init__(self, args):
        super(Random, self).__init__()
        self.args = args
    
    def forward(self, x_support, x_query):
        
        num_sup_samples = len(x_support)
        
        num_q_samples = len(x_query)
        
        scores  = torch.randn(num_sup_samples * num_q_samples)
        
        return scores
        
    def predict(self,  x_support, x_query, y_support):
        
        num_sup_samples = len(x_support)
        
        num_q_samples = len(x_query)
        
        scores  = self.forward(x_support, x_query)
        
        scores = scores.view(num_q_samples, num_sup_samples)
        
        max_score_idx = torch.argmax(scores, dim=1)
       
        max_score_idx = max_score_idx.tolist()
        
        y_q_hat = [y_support[idx] for idx in max_score_idx]
        
        return y_q_hat
    
class MBEMBC(nn.Module):
    def __init__(self, args):
        super(MBEMBC, self).__init__()
        '''
        MBEMBC: Metric-based BiEncoder + Metric-based BinaryClassifier
        '''
        self.args = args
        
        self.bert = BERTEncoder(args.model_name_or_path, sent_embed_type=args.pooling)

        self.encoder = BiEncoder(encoder=self.bert, max_seq_length=args.max_seq_length)
        
        # self.classifier = BinaryClassifier(args, feat_dim = self.encoder.hidden_size)
        
        self.sigmoid = nn.Sigmoid()
            
        
    def forward(self, x_support, x_query):

        interplay_vector = self.encoder.predict(x_query=x_query, x_support=x_support)
        
        scores = interplay_vector
        
        scores = self.sigmoid(scores)
        
        return scores
    
    def predict(self, x_support, x_query, y_support):
        
        num_sup_samples = len(x_support)
        
        num_q_samples = len(x_query)
        
        scores  = self.forward(x_support, x_query)
        
        scores = scores.view(num_q_samples, num_sup_samples)
        
        max_score_idx = torch.argmax(scores, dim=1)
       
        max_score_idx = max_score_idx.tolist()
        
        y_q_hat = [y_support[idx] for idx in max_score_idx]
        
        return y_q_hat
        