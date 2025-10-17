#!/usr/bin/env python3

"""Collection of useful modules."""

import torch.nn as nn


class PrevActionEmbedding(nn.Module):
    def __init__(self, action_space, embedding_size=32):
        super().__init__()
        self.embedding = nn.Embedding(action_space.n + 1, embedding_size)
        self.embedding_size = embedding_size

    @property
    def output_size(self):
        return self.embedding_size

    def forward(self, prev_actions):
        return self.embedding(prev_actions)
