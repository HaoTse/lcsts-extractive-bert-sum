import os

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from torch.nn.init import xavier_uniform_

from summarizer.encoder import Classifier


class Bert(nn.Module):
    def __init__(self, bert_model, cache_dir, load_pretrained_bert, bert_config):
        super(Bert, self).__init__()

        if(load_pretrained_bert):
            self.model = BertModel.from_pretrained(bert_model, cache_dir=cache_dir)
        else:
            self.model = BertModel(bert_config)

    def forward(self, x, segs, mask):
        encoded_layers, _ = self.model(x, segs, attention_mask =mask)
        top_vec = encoded_layers[-1]
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, args, device, load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.args = args
        self.device = device

        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
        self.bert = Bert(args.bert_model, cache_dir, load_pretrained_bert, bert_config)
        self.encoder = Classifier(self.bert.model.config.hidden_size)
        self.config = self.bert.model.config

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

    def load_cp(self, pt):
        self.load_state_dict(pt, strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, sentence_range=None):

        top_vec = self.bert(x, segs, mask) # [batch size, sequence length, hidden size]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss] # [batch size, clss length, hidden size]
        sents_vec = sents_vec * mask_cls[:, :, None].float() # [batch size, clss length, hidden size]
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1) # [batch size, clss length, 1]
        return sent_scores, mask_cls
