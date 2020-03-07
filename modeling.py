import numpy as np
import time

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class SimpleClassifier(nn.Module):
    def __init__(self, config):
        super(SimpleClassifier, self).__init__()
        # encoder: No encoder inside this class.
        # classifier: Do not follow the BERT built-in classifier anymore.
        self.input_dim = 4*config['enc_dim']
        self.drop = nn.Dropout(config['dropout_prob'])
        self.num_labels = config['n_classes']
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, config['fc_dim']),
            nn.Linear(config['fc_dim'], self.num_labels)
        )

    def forward(self, s1_emb, s2_emb, labels=None):
        """
        compute the loss or logits of 2 input sentences.
        """
        u = s1_emb
        v = s2_emb
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        logits = self.classifier(features)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits