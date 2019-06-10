# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF
from .sampled_softmax_loss import SampledSoftmaxLoss


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.use_crf_sl = data.use_crf_sl
        self.use_crf_lm = data.use_crf_lm
        print("build network...")

        self.bilstm_flag = data.HP_bilstm
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss

        # add two more label for down layer lstm, use original label size for CRF
        source_label_size = data.source_label_alphabet_size
        target_label_size = data.target_label_alphabet_size
        if self.use_crf_sl:
            data.source_label_alphabet_size += 2
            data.target_label_alphabet_size += 2

        self.word_hidden = WordSequence(data)

        if self.bilstm_flag:
            self.lstm_hidden = data.HP_hidden_dim // 2
        else:
            self.lstm_hidden = data.HP_hidden_dim

        self.source_hidden2tag = nn.Linear(data.HP_hidden_dim, data.source_label_alphabet_size)
        self.target_hidden2tag = nn.Linear(data.HP_hidden_dim, data.target_label_alphabet_size)

        self.source_lm_soft_max = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden,
                                                     num_samples=data.LM_sample_num, sparse=False, gpu=self.gpu)
        self.target_lm_soft_max = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden,
                                                     num_samples=data.LM_sample_num, sparse=False, gpu=self.gpu)

        self.target_lm_soft_max = self.source_lm_soft_max

        if self.use_crf_sl:
            self.source_crf = CRF(source_label_size, self.gpu)
            self.target_crf = CRF(target_label_size, self.gpu)

        if self.gpu:
            self.source_hidden2tag = self.source_hidden2tag.cuda()
            self.target_hidden2tag = self.target_hidden2tag.cuda()
            self.source_lm_soft_max = self.source_lm_soft_max.cuda()
            self.target_lm_soft_max = self.source_lm_soft_max.cuda()

    def loss(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
             batch_label, lm_seq_tensor, mask):

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        if self.bilstm_flag:
            outs_forward, outs_backward, outs = self.word_hidden(mode, word_inputs, word_seq_lengths, char_inputs,
                                                                 char_seq_lengths, char_seq_recover)

        loss, perplexity, tag_seq = None, None, None
        if mode == 'model1' or mode == 'model3':
            loss, perplexity = self.lm_loss(mode, outs_forward, outs_backward, batch_size, lm_seq_tensor[0],
                                            lm_seq_tensor[1], mask)
        else:
            loss, tag_seq = self.neg_log_likelihood_loss(mode, outs, batch_size, seq_len, batch_label, mask)

        return loss, perplexity, tag_seq

    def neg_log_likelihood_loss(self, mode, outs, batch_size, seq_len, batch_label, mask):

        if mode == 'model2':
            outs = self.source_hidden2tag(outs)
        else:
            outs = self.source_hidden2tag(outs)

        if self.use_crf_sl:
            if mode == 'model2':
                crf = self.source_crf
            else:
                crf = self.source_crf

            total_loss = crf.neg_log_likelihood_loss(outs, mask, batch_label)

            scores, tag_seq = crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        if self.average_batch:
            if batch_size != 0:
                total_loss = total_loss / batch_size
            else:
                total_loss = 0
        return total_loss, tag_seq

    def lm_loss(self, mode, outs_forward, outs_backward, batch_size, lm_forward_seq_tensor, lm_backward_seq_tensor,
                mask):

        if self.bilstm_flag:

            if mode == 'model1':
                soft_max_lm = self.source_lm_soft_max
            else:
                soft_max_lm = self.target_lm_soft_max

            losses = []
            for embedding, targets in ((outs_forward, lm_forward_seq_tensor), (outs_backward, lm_backward_seq_tensor)):
                non_masked_targets = targets.masked_select(mask) - 1
                non_masked_embedding = embedding.masked_select(mask.unsqueeze(-1)).view(-1, self.lstm_hidden)

                losses.append(soft_max_lm(non_masked_embedding, non_masked_targets))

            total_loss = (losses[0] + losses[1]) / 2

            length_mask = torch.sum(mask.float(), dim=1).float()
            num = length_mask.sum(0).data[0]

            if num:
                perplexity = total_loss / num
            else:
                perplexity = 0.0

            if self.average_batch:
                if batch_size:
                    total_loss = total_loss / batch_size
                else:
                    total_loss = 0

            return total_loss, perplexity

    def forward(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        _, _, outs = self.word_hidden(mode, word_inputs, word_seq_lengths, char_inputs,
                                      char_seq_lengths, char_seq_recover)

        if mode == 'model2':
            outs = self.source_hidden2tag(outs)
        else:
            outs = self.source_hidden2tag(outs)

        if self.use_crf_sl:
            if mode == 'model2':
                crf = self.source_crf
            else:
                crf = self.source_crf
            scores, tag_seq = crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            tag_seq = mask.long() * tag_seq

        return tag_seq

    def decode_nbest(self, mode, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                     char_seq_recover, mask, nbest):
        if not self.use_crf_sl:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        _, _, outs = self.word_hidden(mode, word_inputs, word_seq_lengths, char_inputs,
                                      char_seq_lengths, char_seq_recover)
        # batch_size = word_inputs.size(0)
        # seq_len = word_inputs.size(1)
        if mode == 'model2':
            outs = self.source_hidden2tag(outs)
        else:
            outs = self.source_hidden2tag(outs)
        if self.use_crf_sl:
            if mode == 'model2':
                crf = self.source_crf
            else:
                crf = self.source_crf
        scores, tag_seq = crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq
