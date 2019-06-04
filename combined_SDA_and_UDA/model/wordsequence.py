# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .LSTM_base import LSTM
from .ParamGenerator import LSTMParamGenerator


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.drop_lstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.word_rep = WordRep(data)
        self.input_size = data.word_emb_dim

        self.task_num = data.task_number
        self.domain_num = data.domain_number
        self.task_emb_size = data.task_emb_dim
        self.domain_emb_size = data.domain_emb_dim

        self.ner_1_task_id = data.task_alphabet.get_index(data.ner_task_name)
        self.ner_1_domain_id = data.domain_alphabet.get_index(data.domain_1_name)
        self.ner_2_task_id = data.task_alphabet.get_index(data.ner_task_name)
        self.ner_2_domain_id = data.domain_alphabet.get_index(data.domain_2_name)
        self.lm_1_task_id = data.task_alphabet.get_index(data.lm_task_name)
        self.lm_1_domain_id = data.domain_alphabet.get_index(data.domain_1_name)
        self.lm_2_task_id = data.task_alphabet.get_index(data.lm_task_name)
        self.lm_2_domain_id = data.domain_alphabet.get_index(data.domain_2_name)

        if self.use_char:
            self.input_size += data.HP_char_hidden_dim

        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.LSTM_param_generator = LSTMParamGenerator(self.input_size, lstm_hidden, data)

        self.lstm = LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                         bidirectional=self.bilstm_flag)

        if self.gpu:
            self.drop_lstm = self.drop_lstm.cuda()
            self.lstm = self.lstm.cuda()
            self.LSTM_param_generator = self.LSTM_param_generator.cuda()

    def forward(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover):
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
        word_represent = self.word_rep(mode, word_inputs, word_seq_lengths, char_inputs,
                                       char_seq_lengths, char_seq_recover)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        if mode == 'ner1':
            task_idx = self.ner_1_task_id
            domain_idx = self.ner_1_domain_id
        elif mode == 'ner2':
            task_idx = self.ner_2_task_id
            domain_idx = self.ner_2_domain_id
        elif mode == 'lm1':
            task_idx = self.lm_1_task_id
            domain_idx = self.lm_1_domain_id
        elif mode == 'lm2':
            task_idx = self.lm_2_task_id
            domain_idx = self.lm_2_domain_id

        lstm_param = self.LSTM_param_generator(task_idx, domain_idx)
        if 'ner' in mode:
            lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            feature_out = self.drop_lstm(lstm_out)
            outputs_forward, outputs_backward = feature_out.chunk(2, -1)
            outputs = torch.cat((outputs_forward, outputs_backward), -1)
        else:
            lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            outputs = self.drop_lstm(lstm_out)
            outputs_forward, outputs_backward = outputs.chunk(2, -1)

        return outputs_forward, outputs_backward, outputs
