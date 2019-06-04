# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
from __future__ import print_function
import time
import os
import argparse
import random
import math
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from utils import conlleval
from model.seqmodel import SeqModel
from utils.data import Data

try:
    import cPickle as pickle
except ImportError:
    import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_init_supervised(supervised_data):
    print('data_init_supervised')
    supervised_data.build_language_model_alphabet(supervised_data.supervised_lm_1_train,
                                                  supervised_data.supervised_lm_2_train)
    supervised_data.filter_word_count()

    supervised_data.build_alphabet(supervised_data.supervised_ner_1_train, supervised_data.supervised_ner_2_train)
    supervised_data.build_alphabet(supervised_data.supervised_ner_1_dev, supervised_data.supervised_ner_2_dev)
    supervised_data.build_alphabet(supervised_data.supervised_ner_1_test, supervised_data.supervised_ner_2_test)

    supervised_data.build_task_domain_alphabet()
    supervised_data.fix_alphabet()


def data_init_transfer(transfer_data):
    print('data_init_transfer')
    transfer_data.build_language_model_alphabet(transfer_data.transfer_lm_1_train, transfer_data.transfer_lm_2_train)
    transfer_data.filter_word_count()

    transfer_data.build_alphabet(transfer_data.transfer_ner_1_train, None, single_label_alphabet=True)
    transfer_data.build_alphabet(transfer_data.transfer_ner_1_dev, transfer_data.transfer_ner_2_dev,
                                 single_label_alphabet=True)
    transfer_data.build_alphabet(transfer_data.transfer_ner_1_test, transfer_data.transfer_ner_2_test,
                                 single_label_alphabet=True)

    transfer_data.build_task_domain_alphabet()
    transfer_data.fix_alphabet()


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
        pred_label.extend(pred)
        gold_label.extend(gold)
    return pred_label, gold_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(mode, data_instance, label_alphabet, data, model):
    model.eval()
    pred_results = []
    gold_results = []
    for batch_id in range(len(data_instance) // data.HP_batch_size + 1):
        instance = data_instance[batch_id * data.HP_batch_size: (batch_id + 1) * data.HP_batch_size \
            if (batch_id + 1) * data.HP_batch_size < len(data_instance) else len(data_instance)]

        if not instance:
            continue
        instance_batch_data = batchify_with_label(instance, data.HP_gpu, True)

        tag_seq = model(mode, instance_batch_data[0], instance_batch_data[1], instance_batch_data[3],
                        instance_batch_data[4], instance_batch_data[5], instance_batch_data[8])

        pred_label, gold_label = recover_label(tag_seq, instance_batch_data[6], instance_batch_data[8],
                                               label_alphabet, instance_batch_data[2])
        pred_results += pred_label
        gold_results += gold_label
    p, r, f = conlleval.evaluate(gold_results, pred_results, verbose=False)
    print("precision {0}, recall {1}, f1 {2}".format(p, r, f))
    return p, r, f


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
    word_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    lm_forward_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    lm_backward_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()

    mask = Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        if seq_len > 1:
            lm_forward_seq_tensor[idx, 0: seq_len - 1] = word_seq_tensor[idx, 1: seq_len]
            lm_forward_seq_tensor[idx, seq_len - 1] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 1: seq_len] = word_seq_tensor[idx, 0: seq_len - 1]
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        else:
            lm_forward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.Tensor([1] * seq_len)

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    lm_forward_seq_tensor = lm_forward_seq_tensor[word_perm_idx]
    lm_backward_seq_tensor = lm_backward_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
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


def train(train_data):
    print("Training model...")
    train_data.show_data_summary()
    save_data_name = train_data.init_dir + ".init"
    train_data.save(save_data_name)
    model = SeqModel(train_data)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    if train_data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_data.HP_lr,
                              momentum=train_data.HP_momentum, weight_decay=train_data.HP_l2)
    elif train_data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=train_data.HP_lr, weight_decay=train_data.HP_l2)
    elif train_data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=train_data.HP_lr, weight_decay=train_data.HP_l2)
    elif train_data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=train_data.HP_lr, weight_decay=train_data.HP_l2)
    elif train_data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=train_data.HP_lr, weight_decay=train_data.HP_l2)
    else:
        print("Optimizer illegal: %s" % train_data.optimizer)
        exit(1)

    best_dev = -10
    dev_f = []
    test_f = []
    best_epoch = 0

    for idx in range(train_data.HP_iteration):
        epoch_start = time.time()
        print("Epoch: %s/%s" % (idx, train_data.HP_iteration))
        if train_data.optimizer.lower() == "sgd":
            optimizer = lr_decay(optimizer, idx, train_data.HP_lr_decay, train_data.HP_lr)

        random.shuffle(train_data.ner_1_train_idx)
        random.shuffle(train_data.ner_2_train_idx)
        random.shuffle(train_data.lm_1_idx)
        random.shuffle(train_data.lm_2_idx)

        model.train()
        model.zero_grad()

        ner_1_loss = 0
        ner_2_loss = 0
        lm_1_perplexity = 0
        lm_2_perplexity = 0

        ner_1_batch_size = train_data.HP_batch_size
        batch_nums = len(train_data.ner_1_train_idx) // ner_1_batch_size + 1
        ner_2_batch_size = len(train_data.ner_2_train_idx) // batch_nums
        lm_1_batch_size = len(train_data.lm_1_idx) // batch_nums
        lm_2_batch_size = len(train_data.lm_2_idx) // batch_nums

        print("batch size: ", ner_1_batch_size, ner_2_batch_size, lm_1_batch_size, lm_2_batch_size)

        for batch_id in range(batch_nums):
            ner_1_data = train_data.ner_1_train_idx[batch_id * ner_1_batch_size: (batch_id + 1) * ner_1_batch_size if\
            (batch_id + 1) * ner_1_batch_size < len(train_data.ner_1_train_idx) else len(train_data.ner_1_train_idx)]
            ner_2_data = train_data.ner_2_train_idx[batch_id * ner_2_batch_size: (batch_id + 1) * ner_2_batch_size if\
            (batch_id + 1) * ner_2_batch_size < len(train_data.ner_2_train_idx) else len(train_data.ner_2_train_idx)]
            lm_1_data = train_data.lm_1_idx[batch_id * lm_1_batch_size: (batch_id + 1) * lm_1_batch_size if \
                (batch_id + 1) * lm_1_batch_size < len(train_data.lm_1_idx) else len(train_data.lm_1_idx)]
            lm_2_data = train_data.lm_2_idx[batch_id * lm_2_batch_size: (batch_id + 1) * lm_2_batch_size if \
                (batch_id + 1) * lm_2_batch_size < len(train_data.lm_2_idx) else len(train_data.lm_2_idx)]

            ner_1_batch_data = batchify_with_label(ner_1_data, train_data.HP_gpu)
            if train_data.mode == 'supervised':
                ner_2_batch_data = batchify_with_label(ner_2_data, train_data.HP_gpu)
            lm_1_batch_data = batchify_with_label(lm_1_data, train_data.HP_gpu)
            lm_2_batch_data = batchify_with_label(lm_2_data, train_data.HP_gpu)

            losses = []
            perplexities = []

            # word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths,
            # char_seq_recover,  label_seq_tensor, lm_seq_tensor, mask
            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                model.loss('ner1', ner_1_batch_data[0], ner_1_batch_data[1], ner_1_batch_data[3], ner_1_batch_data[4],
                           ner_1_batch_data[5], ner_1_batch_data[6], ner_1_batch_data[7], ner_1_batch_data[8])
            losses.append(loss)

            if train_data.mode == 'supervised':
                loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                    model.loss('ner2', ner_2_batch_data[0], ner_2_batch_data[1], ner_2_batch_data[3],
                               ner_2_batch_data[4], ner_2_batch_data[5], ner_2_batch_data[6], ner_2_batch_data[7],
                               ner_2_batch_data[8])
                losses.append(loss)

            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                model.loss('lm1', lm_1_batch_data[0], lm_1_batch_data[1], lm_1_batch_data[3], lm_1_batch_data[4],
                           lm_1_batch_data[5], lm_1_batch_data[6], lm_1_batch_data[7], lm_1_batch_data[8])
            losses.append(loss)
            perplexities.append(perplexity)

            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = \
                model.loss('lm2', lm_2_batch_data[0], lm_2_batch_data[1], lm_2_batch_data[3], lm_2_batch_data[4],
                           lm_2_batch_data[5], lm_2_batch_data[6], lm_2_batch_data[7], lm_2_batch_data[8])
            losses.append(loss)
            perplexities.append(perplexity)

            model_loss = 0
            loss_rate = [0.8, 1, 0.5, 0.5] if train_data.mode == 'supervised' else [1, 1, 1]
            for loss_id in range(len(losses)):
                model_loss += losses[loss_id] * loss_rate[loss_id]
            model_loss.backward()
            optimizer.step()
            model.zero_grad()

            ner_1_loss += losses[0].data[0]
            ner_2_loss += losses[1].data[0]
            lm_1_perplexity += perplexities[0].data[0]
            lm_2_perplexity += perplexities[1].data[0]

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        print("Epoch: %s training finished. Time: %.2fs." % (idx, epoch_cost))
        print("ner 1 total loss: %s" % ner_1_loss)
        if train_data.mode == 'supervised':
            print("ner 2 total loss: %s" % ner_2_loss)
        print("lm 1 perplexity: %.4f" % math.exp(lm_1_perplexity / batch_nums))
        print("lm 2 perplexity: %.4f" % math.exp(lm_2_perplexity / batch_nums))

        if ner_1_loss > 1e8 or str(ner_1_loss) == "nan" or ner_2_loss > 1e8 or str(ner_2_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)

        evaluate('ner1', train_data.ner_1_dev_idx, train_data.label_alphabet_ner_1, train_data, model)
        if train_data.mode == 'supervised':
            p, r, f = evaluate('ner2', train_data.ner_2_dev_idx, train_data.label_alphabet_ner_2, train_data, model)
        else:
            p, r, f = evaluate('ner2', train_data.ner_2_dev_idx, train_data.label_alphabet_ner_1, train_data, model)
        dev_f.append(f)

        if f > best_dev:
            best_epoch = idx
            print("Exceed previous best f score:", best_dev)
            model_name = train_data.model_dir + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = f

        evaluate('ner1', train_data.ner_1_test_idx, train_data.label_alphabet_ner_1, train_data, model)
        if train_data.mode == 'supervised':
            p, r, f = evaluate('ner2', train_data.ner_2_test_idx, train_data.label_alphabet_ner_2, train_data, model)
        else:
            p, r, f = evaluate('ner2', train_data.ner_2_test_idx, train_data.label_alphabet_ner_1, train_data, model)
        test_f.append(f)
    print("the best dev score is in epoch %s, dev:%.4f, test:%.4f" % (best_epoch, dev_f[best_epoch],
                                                                      test_f[best_epoch]))


def load_model_decode(data):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    model.load_state_dict(torch.load(data.load_model_dir))
    evaluate(data.ner_2_test_idx, data, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cross ner via cross language model')
    parser.add_argument('--config', help='configuration File')
    args = parser.parse_args()

    cross_data = Data()
    cross_data.HP_gpu = torch.cuda.is_available()

    cross_data.read_config(args.config)
    status = cross_data.status.lower()

    print("Seed num:", seed_num)

    if status == 'train':
        print("MODEL: train")

        transfer_flag = False
        if cross_data.mode == 'supervised':
            data_init_supervised(cross_data)
        elif cross_data.mode == 'transfer':
            data_init_transfer(cross_data)
            transfer_flag = True

        cross_data.generate_instance('train', transfer_flag)
        cross_data.generate_instance('dev', transfer_flag)
        cross_data.generate_instance('test', transfer_flag)
        cross_data.build_pretrain_emb()
        train(cross_data)

    elif status == 'decode':
        print("MODEL: decode")
        cross_data.load(cross_data.init_dir)
        cross_data.read_config(args.config)
        cross_data.show_data_summary()
        load_model_decode(cross_data)
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
