# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 14:50:58
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from .wordrep import WordRep
from .LSTM_base import LSTM
from .CPG import Network_param_generater

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim

        self.task_num = data.task_number
        self.domain_num = data.domain_number
        self.task_emb_size = data.task_emb_dim #task embdding
        self.domain_emb_size = data.domain_emb_dim #domain_embedding
        self.pretrain_task_emb = None
        self.pretrain_domain_emb = None
        self.model1_task_idx = data.task_alphabet.get_index(data.model1_task_name)
        self.model1_domain_idx = data.domain_alphabet.get_index(data.model1_domain_name)
        self.model2_task_idx = data.task_alphabet.get_index(data.model2_task_name)
        self.model2_domain_idx = data.domain_alphabet.get_index(data.model2_domain_name)
        self.model3_task_idx = data.task_alphabet.get_index(data.model3_task_name)
        self.model3_domain_idx = data.domain_alphabet.get_index(data.model3_domain_name)
        self.model4_task_idx = data.task_alphabet.get_index(data.model4_task_name)
        self.model4_domain_idx = data.domain_alphabet.get_index(data.model4_domain_name)

        #self.LM_flag = data.HP_LM

        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.word_feature_extractor = data.word_feature_extractor

        self.LSTM_param_generator = Network_param_generater(self.input_size, lstm_hidden, data)

        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            # lstm_param = self.LSTM_param_generator(self.task_idx, self.domain_idx)
            self.lstm = LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "CNN":
            # cnn_hidden = data.HP_hidden_dim
            self.word2cnn = nn.Linear(self.input_size, data.HP_hidden_dim)
            self.cnn_layer = data.HP_cnn_layer
            print("CNN layer: ", self.cnn_layer)
            self.cnn_list = nn.ModuleList()
            self.cnn_drop_list = nn.ModuleList()
            self.cnn_batchnorm_list = nn.ModuleList()
            kernel = 3
            pad_size = (kernel-1)/2
            for idx in range(self.cnn_layer):
                self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size=kernel, padding=pad_size))
                self.cnn_drop_list.append(nn.Dropout(data.HP_dropout))
                self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))
        # The linear layer that maps from hidden state space to tag space
        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            if self.word_feature_extractor == "CNN":
                self.word2cnn = self.word2cnn.cuda()
                for idx in range(self.cnn_layer):
                    self.cnn_list[idx] = self.cnn_list[idx].cuda()
                    self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
                    self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
            else:
                self.lstm = self.lstm.cuda()
                self.LSTM_param_generator = self.LSTM_param_generator.cuda()

        # model.



    def forward(self, mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_represent = self.wordrep(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## word_embs (batch_size, seq_len, embed_size)
        if self.word_feature_extractor == "CNN":
            word_in = F.tanh(self.word2cnn(word_represent)).transpose(2,1).contiguous()
            for idx in range(self.cnn_layer):
                if idx == 0:
                    cnn_feature = F.relu(self.cnn_list[idx](word_in))
                else:
                    cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
                cnn_feature = self.cnn_drop_list[idx](cnn_feature)
                cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
            feature_out = cnn_feature.transpose(2,1).contiguous()
        else:
            packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
            hidden = None
            if mode == 'model1':
                task_idx = self.model1_task_idx
                domain_idx = self.model1_domain_idx
            elif mode == 'model2':
                task_idx = self.model2_task_idx
                domain_idx = self.model2_domain_idx
            elif mode == 'model3':
                task_idx = self.model3_task_idx
                domain_idx = self.model3_domain_idx
            elif mode == 'model4':
                task_idx = self.model4_task_idx
                domain_idx = self.model4_domain_idx

            lstm_param = self.LSTM_param_generator(task_idx, domain_idx)
            outputs_forward, outputs_backward, outputs = None, None, None
            # print('+++++++++++++')
            # print(len(lstm_param))
            ###Language model
            if mode == 'model1' or mode == 'model3':
                lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
                lstm_out, _ = pad_packed_sequence(lstm_out)
                ## lstm_out (seq_len, seq_len, hidden_size)
                feature_out = self.droplstm(lstm_out.transpose(1,0))
            ## feature_out (batch_size, seq_len, hidden_size)
                outputs_forward, outputs_backward = feature_out.chunk(2, -1)

            else:
            ### sequence labeling
                lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
                lstm_out, _ = pad_packed_sequence(lstm_out)
            ## lstm_out (seq_len, seq_len, hidden_size)
                outputs = self.droplstm(lstm_out.transpose(1,0))

            return outputs_forward, outputs_backward, outputs