import collections
import itertools
import logging
import pdb
import warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

warnings.simplefilter('ignore')

#device = "cpu"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Adapt from `https://github.com/tdopierre/ProtAugment/blob/main/utils/math.py`
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    # [n, 1, d] -> [n, m, d]
    x = x.unsqueeze(1).expand(n, m, d)
    # [1, m, d] -> [n, m, d]
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


# Adapt from `https://github.com/tdopierre/ProtAugment/blob/main/utils/math.py`
def cosine_similarity(x, y):
    x = (x / x.norm(dim=1).view(-1, 1))
    y = (y / y.norm(dim=1).view(-1, 1))

    return x @ y.T


def merge_embeds(z, offsets):
    """ Get mean embeddings from offset
    """
    # [batch_size, seq_len]
    _start, _end, ranges = torch.broadcast_tensors(
        offsets[0].unsqueeze(-1),
        offsets[1].unsqueeze(-1),
        torch.arange(0, z.shape[0]).long().to(device)
    )
    idx = (torch.ge(ranges, _start) &
           torch.lt(ranges, _end)).float().to(device)
    # [batch_size, dim]
    merged_z = torch.div(
        torch.matmul(idx, z),
        torch.sum(idx, dim=1).unsqueeze(-1)
    )
    return merged_z


class BERTEncoder(nn.Module):
    def __init__(self, config_name_or_path, sent_embed_type='cls'):
        super(BERTEncoder, self).__init__()
        logger.info(f"Loading PLM @ {config_name_or_path} ")
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.bert_config = AutoConfig.from_pretrained(config_name_or_path)
        self.bert = AutoModel.from_pretrained(config_name_or_path, config=self.bert_config).to(device)
        logger.info(f"PLM is loaded.")
        self.hidden_size = self.bert_config.hidden_size
        self.sent_embed_type = sent_embed_type
        
    def get_sent_embeddings(self, batch):
        fw = self.bert.forward(**batch)
        # pooler([CLS])
        if self.sent_embed_type == 'cls':
            return fw.pooler_output
            # [CLS] 
            # return fw.last_hidden_state[:, 0]
        elif self.sent_embed_type == 'avg':
            # get sentence embeddings by averaging token embeddings
            sentence_embeddings = torch.bmm(batch["attention_mask"].unsqueeze(1), fw.last_hidden_state).squeeze(1)
            sentence_embeddings /= batch["attention_mask"].sum(1, keepdim=True)
            return sentence_embeddings
        else:
            raise NotImplementedError


class BiEncoder(nn.Module):
    def __init__(self, encoder, max_seq_length=64):
        super(BiEncoder, self).__init__()
        self.encoder = encoder
        self.max_seq_length = max_seq_length
        # self.hidden_size = 3 * self.encoder.hidden_size
        self.hidden_size = 2 * self.encoder.hidden_size
        logger.info(f"BiEncoder loaded.")

    def process_episode(self, x_query, x_support):
        n_query = len(x_query)
        n_support = len(x_support)
        x = x_query + x_support
        x_batch = self.encoder.tokenizer(
            x,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length"
        )
        x_batch = {k: v.to(device) for k, v in x_batch.items()}

        return x_batch, (n_support, n_query)

    def forward(self, x_query, x_support):
        """
            x_query           : [nq] list of query samples
            x_support         : [ns] list of support samples
        return
            hidden matrix     : [nq * ns]
        """
        # Process episode
        x_batch, (n_support, n_query) = self.process_episode(x_query, x_support)

        # Compute vectors
        z = self.encoder.get_sent_embeddings(x_batch)
        assert torch.isnan(z).sum() == 0, "z has nan"
        # Dispatch
        z_query = z[:n_query]
        z_support = z[n_query:]

        # [n, 1, d] -> [n, m, d]
        z_query = z_query.unsqueeze(1).expand(n_query, n_support, -1)
        # [1, m, d] -> [n, m, d]
        z_support = z_support.unsqueeze(0).expand(n_query, n_support, -1)
        # [n, m, d]
        # concat_z = torch.cat([z_query, z_support, z_query - z_support], 2)
        concat_z = torch.cat([torch.abs(z_query - z_support), z_query * z_support], 2)
        # concat_z = self.mlp(torch.cat([z_query, z_support], 2))
        return concat_z.view(n_query * n_support, self.hidden_size)

    def predict(self, x_query, x_support):
        # Process episode
        x_batch, (n_support, n_query) = self.process_episode(x_query, x_support)

        # Compute vectors
        z = self.encoder.get_sent_embeddings(x_batch)
        assert torch.isnan(z).sum() == 0, "z has nan"
        # Dispatch
        z_query = z[:n_query]
        z_support = z[n_query:]
        return cosine_similarity(z_query, z_support).view(-1, 1)

class CrossEncoder(nn.Module):
    def __init__(self, encoder, max_seq_length=64, batch_size=32):
        super(CrossEncoder, self).__init__()
        self.encoder = encoder
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.hidden_size = self.encoder.hidden_size
        logger.info(f"CrossEncoder loaded.")

    def process_episode(self, x_query, x_support):
        """ process episode
        
            x_query   : [nq] list of query samples
            x_support : [ns] list of support samples
        return
            batch     : [nq * ns]
        """
        xq = []
        xs = []
        # loop over query
        for q in x_query:
            for s in x_support:
                xq.append(q)
                xs.append(s)

        org_x_batch = self.encoder.tokenizer(
            xq,
            xs,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length"
        )
        org_x_batch = {k: v.to(device) for k, v in org_x_batch.items()}
        return org_x_batch

    def forward(self, x_query, x_support):
        """
            x_query           : [nq] list of query samples
            x_support         : [ns] list of support samples
        return
            hidden matrix     : [nq * ns]
        """
        x_batch = self.process_episode(x_query, x_support)
        z = self.encoder.get_sent_embeddings(x_batch)
        return z
