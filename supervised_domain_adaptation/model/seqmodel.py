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
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf sl: ", self.use_crf_sl)
        self.bilstm_flag = data.HP_bilstm
        # self.LM_flag = data.HP_LM
        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        self.LM_use_sample_softmax = data.LM_use_sample
        ## add two more label for downlayer lstm, use original label size for CRF
        label_size_1 = data.label_alphabet_1_size
        label_size_2 = data.label_alphabet_2_size
        if self.use_crf_sl:
            data.label_alphabet_1_size += 2
            data.label_alphabet_2_size += 2
        self.word_hidden = WordSequence(data)
        if self.bilstm_flag:
            self.lstm_hidden = data.HP_hidden_dim // 2
        else:
            self.lstm_hidden = data.HP_hidden_dim
        self.hidden2tag_sl_1 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_1_size)
        self.hidden2tag_sl_2 = nn.Linear(data.HP_hidden_dim, data.label_alphabet_2_size)
        if self.LM_use_sample_softmax:
            self._LM_softmax = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden, num_samples=data.LM_sample_num, sparse=False, gpu=self.gpu)
        else:
            self._LM_softmax = LM_softmax(self.lstm_hidden, data.word_alphabet_size, gpu=self.gpu)
        if self.use_crf_sl:
            self.crf_2 = CRF(label_size_1, self.gpu)
            self.crf_4 = CRF(label_size_2, self.gpu)

        if self.gpu:
            self.hidden2tag_sl_1 = self.hidden2tag_sl_1.cuda()
            self.hidden2tag_sl_2 = self.hidden2tag_sl_2.cuda()
            self._LM_softmax = self._LM_softmax.cuda()

    def loss(self, mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, lm_seq_tensor, mask):
        if self.bilstm_flag:
            outs_forward, outs_backward, outs = self.word_hidden(mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size= word_inputs.size(0)
        seq_len = word_inputs.size(1)
        loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = None, None, None, None, None
        if mode == 'model1' or mode == 'model3':
            loss, perplexity, tag_seq_forward, tag_seq_backward = self.LM_loss(mode, outs_forward, outs_backward, batch_size, seq_len, lm_seq_tensor[0], lm_seq_tensor[1], mask)
        else:
            loss, tag_seq = self.neg_log_likelihood_loss(mode, outs, batch_size, seq_len, batch_label, mask)

        return loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq

    def neg_log_likelihood_loss(self, mode, outs, batch_size, seq_len, batch_label, mask):
        if mode == 'model2':
            outs = self.hidden2tag_sl_1(outs)
        else:
            outs = self.hidden2tag_sl_2(outs)
        if self.use_crf_sl:
            if mode == 'model2':
                crf = self.crf_2
            else:
                crf = self.crf_4
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

    def LM_loss(self, mode, outs_forward, outs_backward, batch_size, seq_len, lm_forward_seq_tensor, lm_backward_seq_tensor, mask):

        if self.bilstm_flag:

                softmax_lm = self._LM_softmax
                if self.LM_use_sample_softmax:
                    losses = []
                    for idx, embedding, targets in ((0, outs_forward, lm_forward_seq_tensor),
                                                    (1, outs_backward, lm_backward_seq_tensor)):
                        non_masked_targets = targets.masked_select(mask) - 1
                        non_masked_embedding = embedding.masked_select(
                            mask.unsqueeze(-1)
                        ).view(-1, self.lstm_hidden)
                        # print(non_masked_targets)
                        #losses.append(softmax_lm(mode, non_masked_embedding, non_masked_targets))
                        losses.append(softmax_lm(non_masked_embedding, non_masked_targets))
                    total_loss = 0.5 * (losses[0] + losses[1])
                    tag_seq_forward, tag_seq_backward = None, None
                else:
                    loss_forward, score_forward = softmax_lm(outs_forward, lm_forward_seq_tensor, batch_size, seq_len)
                    loss_backward, score_backward = softmax_lm(outs_backward, lm_backward_seq_tensor, batch_size, seq_len)
                    total_loss = 0.5 * (loss_forward + loss_backward)
                    _, tag_seq_forward = torch.max(score_forward, 1)
                    _, tag_seq_backward = torch.max(score_backward, 1)
                    tag_seq_forward = tag_seq_forward.view(batch_size, seq_len)
                    tag_seq_backward = tag_seq_backward.view(batch_size, seq_len)

                length_mask = torch.sum(mask.float(), dim=1).float()
                # num = length_mask.sum(0).data.numpy()[0]
                num = length_mask.sum(0).data[0]
                if num != 0:
                    perplexity = total_loss / num
                else:
                    perplexity = 0.0
                if self.average_batch:
                    if batch_size != 0:
                        total_loss = total_loss / batch_size
                    else:
                        total_loss = 0

                return total_loss, perplexity, tag_seq_forward, tag_seq_backward




    def forward(self, mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)

        _, _, outs = self.word_hidden(mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                                      char_seq_recover)
        if mode == 'model2':
            outs = self.hidden2tag_sl_1(outs)
        else:
            outs = self.hidden2tag_sl_2(outs)
        if self.use_crf_sl:
            if mode =='model2':
                crf = self.crf_2
            else:
                crf = self.crf_4
            scores, tag_seq = crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq

        return tag_seq

    def decode_nbest(self, mode, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf_sl:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        _, _, outs = self.word_hidden(mode, word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        if mode == 'model2':
            outs = self.hidden2tag_sl_1(outs)
        else:
            outs = self.hidden2tag_sl_2(outs)
        if self.use_crf_sl:
            if mode =='model2':
                crf = self.crf_2
            else:
                crf = self.crf_4
        scores, tag_seq = crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq

class LM_softmax(nn.Module):
    def __init__(self, hidden_size, target_size, gpu=False):
        super(LM_softmax, self).__init__()
        self.hidden_to_tag = nn.Linear(hidden_size, target_size)
        if gpu:
            self.hidden_to_tag = self.hidden_to_tag.cuda()

    def forward(self, embedding, lm_seq_tensor, batch_size, seq_len):
        outputs = self.hidden_to_tag(embedding)
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outputs = outputs.view(batch_size * seq_len, -1)
        scores = F.log_softmax(outputs, 1)
        loss = loss_function(scores, lm_seq_tensor.view(batch_size * seq_len))

        return loss, scores
