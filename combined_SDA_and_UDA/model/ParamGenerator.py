# -*- coding: utf-8 -*-
# @Author: Chen jia
# @Date:   2018-10-15
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
import math
import torch
from torch import nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np


class ParamGenerator(Module):

    def __init__(self, input_size, hidden_size, task_emb_size, domain_emb_size, layer_num=1,
                 bidirectional=False, bias=True):
        super(ParamGenerator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.task_emb_size = task_emb_size
        self.domain_emb_size = domain_emb_size
        self.layer_num = layer_num
        self.bias = bias
        self.bidirectional = bidirectional
        self.direct_num = 2 if bidirectional else 1
        gate_size = 4 * self.hidden_size

        w_task_col_num = 0
        w_task_roll_num = self.task_emb_size
        w_domain_roll_num = self.domain_emb_size
        for idx_layer in range(self.layer_num):
            for direct in range(self.direct_num):
                layer_input_size = self.input_size if idx_layer == 0 else self.hidden_size * self.direct_num
                w_task_col_num += gate_size * layer_input_size
                w_task_col_num += gate_size * self.hidden_size
                if self.bias:
                    w_task_col_num += gate_size
                    w_task_col_num += gate_size

        self.W_task = Parameter(torch.Tensor(w_task_roll_num, w_domain_roll_num, w_task_col_num))
        self.reset_param()

    def reset_param(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, task_emb, domain_emb):
        param_list = []
        param_matrix = torch.matmul(domain_emb.squeeze(0), self.W_task)
        param_matrix = torch.mm(task_emb, param_matrix)
        gate_size = 4 * self.hidden_size
        fund_position = 0
        for idx_layer in range(self.layer_num):
            for direct in range(self.direct_num):
                layer_input_size = self.input_size if idx_layer == 0 else self.hidden_size * self.direct_num
                w_ih = param_matrix[0, fund_position: fund_position + gate_size * layer_input_size].\
                    view(gate_size, layer_input_size)
                fund_position += gate_size * layer_input_size
                w_hh = param_matrix[0, fund_position: fund_position + gate_size * self.hidden_size].\
                    view(gate_size, self.hidden_size)
                fund_position += gate_size * self.hidden_size
                if self.bias:
                    b_ih = param_matrix[0, fund_position: fund_position + gate_size].view(gate_size)
                    fund_position += gate_size
                    b_hh = param_matrix[0, fund_position: fund_position + gate_size].view(gate_size)
                    fund_position += gate_size
                    param_list.append((w_ih, w_hh, b_ih, b_hh))
        return param_list


class LSTMParamGenerator(Module):

    def __init__(self, input_size, hidden_size, data):
        super(LSTMParamGenerator, self).__init__()
        layer_num = data.HP_lstm_layer
        bilstm_flag = data.HP_bilstm
        task_emb_size = data.task_emb_dim
        domain_emb_size = data.domain_emb_dim
        task_num = data.task_alphabet_size
        domain_num = data.domain_alphabet_size
        self.gpu = data.HP_gpu
        self.cpg_ = ParamGenerator(input_size, hidden_size, task_emb_size, domain_emb_size, layer_num,
                                   bilstm_flag, bias=True)
        self.task_emb_vocab = nn.Embedding(task_num, task_emb_size)
        self.domain_emb_vocab = nn.Embedding(domain_num, domain_emb_size)
        self.task_emb_vocab.weight.data.copy_(torch.from_numpy(self.random_embedding(task_num, task_emb_size)))
        self.domain_emb_vocab.weight.data.copy_(torch.from_numpy(self.random_embedding(domain_num, domain_emb_size)))

        if self.gpu:
            self.cpg_ = self.cpg_.cuda()
            self.task_emb_vocab = self.task_emb_vocab.cuda()
            self.domain_emb_vocab = self.domain_emb_vocab.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, task_idx, domain_idx):
        task_id = Variable(torch.LongTensor([task_idx]))
        domain_id = Variable(torch.LongTensor([domain_idx]))
        if self.gpu:
            task_id = task_id.cuda()
            domain_id = domain_id.cuda()
        task_emb = self.task_emb_vocab(task_id)
        domain_emb = self.domain_emb_vocab(domain_id)

        return self.cpg_(task_emb, domain_emb)
