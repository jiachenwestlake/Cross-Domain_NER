# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import argparse
import random
import math
import torch
import gc
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from utils.data import Data

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    import cPickle as pickle
except ImportError:
    import pickle

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    ner_label = ["B-PER", "I-PER", "E-PER", "S-PER",
                 "B-LOC", "I-LOC", "E-LOC", "S-LOC",
                 "B-ORG", "I-ORG", "E-ORG", "S-ORG",
                 "B-MISC", "I-MISC", "E-MISC", "S-MISC", "O"]
    for label in ner_label:
        data.source_label_alphabet.add(label)
        data.target_label_alphabet.add(label)

    data.build_alphabet_lm_raw(data.source_lm_dir)
    data.build_alphabet_lm_raw(data.target_lm_dir)

    data.filter_word_count()

    data.build_alphabet(data.source_train_dir, domain="source")
    data.build_alphabet(data.target_train_dir, domain="target")

    data.build_alphabet(data.source_dev_dir, domain="source")
    data.build_alphabet(data.target_dev_dir, domain="target")

    data.build_alphabet(data.source_test_dir, domain="source")
    data.build_alphabet(data.target_test_dir, domain="target")

    data.build_task_domain_alphabet()
    data.fix_alphabet()

    for i in range(data.source_label_alphabet.size() - 1):
        print(data.source_label_alphabet.instances[i])
    for i in range(data.target_label_alphabet.size() - 1):
        print(data.target_label_alphabet.instances[i])


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, nbest=None):

    if name == "train":
        instances_1 = data.source_train_idx
        instances_2 = data.target_train_idx
    elif name == "dev-test":
        instances_1 = data.source_dev_idx
        instances_2 = data.target_dev_idx
    elif name == 'test':
        instances_1 = data.source_test_idx
        instances_2 = data.target_test_idx
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)

    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()

    nbest_pred_results_1 = []
    pred_scores_1 = []
    pred_results_1 = []
    gold_results_1 = []
    train_num = len(instances_1)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances_1[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, \
        lm_seq_tensor, mask = batchify_with_label(instance, data.HP_gpu, True)

        if nbest:
            scores, nbest_tag_seq = model.decode_nbest('model2', batch_word, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.source_label_alphabet, batch_wordrecover)
            nbest_pred_results_1 += nbest_pred_result
            pred_scores_1 += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq_1 = nbest_tag_seq[:, :, 0]
        else:
            tag_seq_1 = model('model2', batch_word, batch_wordlen, batch_char, batch_charlen,
                              batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq_1, batch_label, mask, data.source_label_alphabet, batch_wordrecover)
        pred_results_1 += pred_label
        gold_results_1 += gold_label
    # decode_time = time.time() - start_time
    # speed = len(instances)/decode_time
    acc_1, p_1, r_1, f_1 = get_ner_fmeasure(gold_results_1, pred_results_1, "BMES")

    nbest_pred_results_2 = []
    pred_scores_2 = []
    pred_results_2 = []
    gold_results_2 = []
    train_num = len(instances_2)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances_2[start:end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label,\
        lm_seq_tensor, mask = batchify_with_label(instance, data.HP_gpu, True)

        if nbest:
            scores, nbest_tag_seq = model.decode_nbest('model4', batch_word, batch_wordlen, batch_char,
                                                       batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.target_label_alphabet, batch_wordrecover)
            nbest_pred_results_2 += nbest_pred_result
            pred_scores_2 += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq_2 = nbest_tag_seq[:, :, 0]
        else:
            tag_seq_2 = model('model4', batch_word, batch_wordlen, batch_char, batch_charlen,
                              batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq_2, batch_label, mask, data.target_label_alphabet, batch_wordrecover)
        pred_results_2 += pred_label
        gold_results_2 += gold_label
    # decode_time = time.time() - start_time
    # speed = len(instances)/decode_time
    acc_2, p_2, r_2, f_2 = get_ner_fmeasure(gold_results_2, pred_results_2, "BMES")
    acc = [acc_1, acc_2]
    p = [p_1, p_2]
    r = [r_1, r_2]
    f = [f_1, f_2]
    pred_results = [pred_results_1, pred_results_2]
    pred_scores = [pred_scores_1, pred_scores_2]
    nbest_pred_results = [nbest_pred_results_1, nbest_pred_results_2]
    decode_time = time.time() - start_time
    speed = (len(instances_1) + len(instances_2)) / decode_time
    if nbest:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores

    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()

    lm_forward_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    lm_backward_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        if seqlen > 1:
            lm_forward_seq_tensor[idx, 0: seqlen - 1] = word_seq_tensor[idx, 1: seqlen]
            lm_forward_seq_tensor[idx, seqlen - 1] = torch.LongTensor([1])  # unk word

            lm_backward_seq_tensor[idx, 1: seqlen] = word_seq_tensor[idx, 0: seqlen - 1]
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        else:
            lm_forward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]

    lm_forward_seq_tensor = lm_forward_seq_tensor[word_perm_idx]
    lm_backward_seq_tensor = lm_backward_seq_tensor[word_perm_idx]

    mask = mask[word_perm_idx]

    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),
                                        volatile=volatile_flag).long()

    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()

        lm_forward_seq_tensor = lm_forward_seq_tensor.cuda()
        lm_backward_seq_tensor = lm_backward_seq_tensor.cuda()

        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    lm_seq_tensor = [lm_forward_seq_tensor, lm_backward_seq_tensor]
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, \
           label_seq_tensor, lm_seq_tensor, mask


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    data.save(save_data_name)

    model = SeqModel(data)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)

    best_dev_f1 = -1
    test_f1 = []
    best_epoch = 0

    # start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)

        instance_count = 0
        total_perplexity_1 = 0
        total_perplexity_2 = 0

        total_loss_1 = 0
        total_loss_2 = 0
        total_loss_3 = 0
        total_loss_4 = 0

        random.shuffle(data.source_train_idx)
        random.shuffle(data.target_train_idx)
        random.shuffle(data.source_lm_idx)
        random.shuffle(data.target_lm_idx)

        model.train()
        model.zero_grad()

        batch_size_1 = data.HP_batch_size
        train_num_1 = len(data.source_train_idx)
        train_num_2 = len(data.target_train_idx)
        train_num_3 = len(data.source_lm_idx)
        train_num_4 = len(data.target_lm_idx)

        batch_num = train_num_1 // batch_size_1 + 1
        batch_size_2 = train_num_2 // batch_num
        batch_size_3 = train_num_3 // batch_num
        batch_size_4 = train_num_4 // batch_num

        for batch_id in range(batch_num):

            instance_1 = data.source_train_idx[batch_id * batch_size_1: (batch_id + 1) * batch_size_1
            if ((batch_id + 1) * batch_size_1) < train_num_1 else train_num_1]
            instance_2 = data.target_train_idx[batch_id * batch_size_2: (batch_id + 1) * batch_size_2
            if ((batch_id + 1) * batch_size_2) < train_num_2 else train_num_2]
            instance_3 = data.source_lm_idx[batch_id * batch_size_3: (batch_id + 1) * batch_size_3
            if ((batch_id + 1) * batch_size_3) < train_num_3 else train_num_3]
            instance_4 = data.target_lm_idx[batch_id * batch_size_4: (batch_id + 1) * batch_size_4
            if ((batch_id + 1) * batch_size_4) < train_num_4 else train_num_4]

            if not instance_1 or not instance_2:
                continue

            # NER
            batch_word_1, batch_wordlen_1, batch_wordrecover_1, batch_char_1, batch_charlen_1, \
            batch_charrecover_1, batch_label_1, lm_seq_tensor_1, mask_1 = batchify_with_label(instance_1, data.HP_gpu)

            batch_word_2, batch_wordlen_2, batch_wordrecover_2, batch_char_2, batch_charlen_2, \
            batch_charrecover_2, batch_label_2, lm_seq_tensor_2, mask_2 = batchify_with_label(instance_2, data.HP_gpu)

            # LM
            batch_word_3, batch_wordlen_3, batch_wordrecover_3, batch_char_3, batch_charlen_3, \
            batch_charrecover_3, batch_label_3, lm_seq_tensor_3, mask_3 = batchify_with_label(instance_3 + instance_1, data.HP_gpu)
            batch_word_4, batch_wordlen_4, batch_wordrecover_4, batch_char_4, batch_charlen_4, \
            batch_charrecover_4, batch_label_4, lm_seq_tensor_4, mask_4 = batchify_with_label(instance_4 + instance_2, data.HP_gpu)

            batch_word = [batch_word_1, batch_word_2, batch_word_3, batch_word_4]
            batch_wordlen = [batch_wordlen_1, batch_wordlen_2, batch_wordlen_3, batch_wordlen_4]

            batch_char = [batch_char_1, batch_char_2, batch_char_3, batch_char_4]
            batch_charlen = [batch_charlen_1, batch_charlen_2, batch_charlen_3, batch_charlen_4]

            batch_charrecover = [batch_charrecover_1, batch_charrecover_2, batch_charrecover_3, batch_charrecover_4]
            batch_label = [batch_label_1, batch_label_2, batch_label_3, batch_label_4]

            lm_seq_tensor = [lm_seq_tensor_1, lm_seq_tensor_2, lm_seq_tensor_3, lm_seq_tensor_4]
            mask = [mask_1, mask_2, mask_3, mask_4]

            instance_count += 1
            loss_ = []
            perplexity_ = []

            # source language model
            loss, perplexity, tag_seq = model.loss('model1', batch_word[2], batch_wordlen[2], batch_char[2],
                                                   batch_charlen[2], batch_charrecover[2], batch_label[2],
                                                   lm_seq_tensor[2], mask[2])
            loss_.append(loss)
            perplexity_.append(perplexity)
            # source NER
            loss, perplexity, tag_seq = model.loss('model2', batch_word[0], batch_wordlen[0], batch_char[0],
                                                   batch_charlen[0], batch_charrecover[0], batch_label[0],
                                                   lm_seq_tensor[0], mask[0])

            loss_.append(loss)
            # target language model
            loss, perplexity, tag_seq = model.loss('model3', batch_word[3], batch_wordlen[3], batch_char[3],
                                                   batch_charlen[3], batch_charrecover[3], batch_label[3],
                                                   lm_seq_tensor[3], mask[3])

            loss_.append(loss)
            perplexity_.append(perplexity)
            loss = 0
            model_num = len(loss_)
            for loss_id in range(model_num):
                loss += loss_[loss_id]
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total_loss_1 += loss_[0].data[0]
            total_loss_2 += loss_[1].data[0]
            total_loss_3 += loss_[2].data[0]

            total_perplexity_1 += perplexity_[0].data[0]
            total_perplexity_2 += perplexity_[1].data[0]


        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        source_lm_perplexity = math.exp(total_perplexity_1 / batch_num)
        target_lm_perplexity = math.exp(total_perplexity_2 / batch_num)

        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, train_num_1 / epoch_cost, total_loss_2))
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total perplexity: %.4f" % (
            idx, epoch_cost, train_num_3 / epoch_cost, source_lm_perplexity))

        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, train_num_2 / epoch_cost, total_loss_4))
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total perplexity: %.4f" % (
            idx, epoch_cost, train_num_4 / epoch_cost, target_lm_perplexity))

        if total_loss_1 > 1e8 or str(total_loss_1) == "nan" or total_loss_2 > 1e8 or str(
                total_loss_2) == "nan" or total_loss_3 > 1e8 or str(total_loss_3) == "nan" or total_loss_4 > 1e8 or str(
            total_loss_4) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        # dev
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev-test")
        test_f1.append(f[1])
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        current_score = f[0]
        print("Dev-Source: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            dev_cost, speed, acc[0], p[0], r[0], f[0]))
        print("Test-Target: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            dev_cost, speed, acc[1], p[1], r[1], f[1]))

        if current_score > best_dev_f1:
            best_epoch = idx
            print("Exceed previous best f score:", best_dev_f1)
            model_name = data.model_dir + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev_f1 = current_score

        if current_score > 0.72:
            print("change optim sgd:")
            optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=data.HP_momentum,
                                  weight_decay=data.HP_l2)
        print("The best Source-domain dev f-score: %.4f, Target-domain f-score: %.4f" % (best_dev_f1, test_f1[best_epoch]))


def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        name, time_cost, speed, acc[0], p[0], r[0], f[0]))
    print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        name, time_cost, speed, acc[1], p[1], r[1], f[1]))
    return pred_results[1], pred_scores[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--config', help='Configuration File')

    args = parser.parse_args()
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    data.read_config(args.config)
    status = data.status.lower()
    print("Seed num:", seed_num)
    print("status :", status)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        data.show_data_summary()
        data.generate_instance('dev')
        decode_results, pred_scores = load_model_decode(data, 'dev')

        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'dev')
        else:
            data.write_decoded_results(decode_results, 'dev')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
