# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from .charbilstm import CharBiLSTM
from .charbigru import CharBiGRU
from .charcnn import CharCNN


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_feature_extractor == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim,
                                            self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim,
                                               self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), self.char_embedding_dim,
                                              self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_feature_extractor"
                      " (CNN/LSTM/GRU/ALL).")
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()

    def forward(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embeds = self.word_embedding(word_inputs)
        word_list = [word_embeds]

        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_list.append(char_features)
        word_represent = self.drop(torch.cat(word_list, 2))
        return word_represent
