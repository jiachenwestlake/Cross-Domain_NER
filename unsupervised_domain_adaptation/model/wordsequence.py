# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 14:50:58
from __future__ import print_function
from __future__ import absolute_import
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from .LSTM_base import LSTM
from .CPG import NetworkParamGenerator


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim

        self.task_num = data.task_number
        self.domain_num = data.domain_number
        self.task_emb_size = data.task_emb_dim  # task embdding
        self.domain_emb_size = data.domain_emb_dim  # domain_embedding

        self.model1_task_idx = data.task_alphabet.get_index(data.model1_task_name)
        self.model1_domain_idx = data.domain_alphabet.get_index(data.model1_domain_name)
        self.model2_task_idx = data.task_alphabet.get_index(data.model2_task_name)
        self.model2_domain_idx = data.domain_alphabet.get_index(data.model2_domain_name)
        self.model3_task_idx = data.task_alphabet.get_index(data.model3_task_name)
        self.model3_domain_idx = data.domain_alphabet.get_index(data.model3_domain_name)
        self.model4_task_idx = data.task_alphabet.get_index(data.model4_task_name)
        self.model4_domain_idx = data.domain_alphabet.get_index(data.model4_domain_name)

        if self.use_char:
            self.input_size += data.HP_char_hidden_dim

        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.LSTM_param_generator = NetworkParamGenerator(self.input_size, lstm_hidden, data)

        self.lstm = LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True,
                         bidirectional=self.bilstm_flag)
        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.lstm = self.lstm.cuda()
            self.LSTM_param_generator = self.LSTM_param_generator.cuda()

    def forward(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
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
        word_represent = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)

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

        if mode == 'model1' or mode == 'model3':
            lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            feature_out = self.droplstm(lstm_out.transpose(1, 0))
            outputs_forward, outputs_backward = feature_out.chunk(2, -1)
        else:
            lstm_out, hidden = self.lstm(packed_words, lstm_param, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            outputs = self.droplstm(lstm_out.transpose(1, 0))

        return outputs_forward, outputs_backward, outputs
