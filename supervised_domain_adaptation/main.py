# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import sys
import os
import argparse
import random
import math
import torch
import gc
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from utils.data import Data

try:
    import cPickle as pickle
except ImportError:
    import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data):
    #data.initial_feature_alphabets()

    data.build_alphabet(data.train_dir_1, data.train_dir_2)
    data.build_alphabet(data.dev_dir_1, data.dev_dir_2)
    data.build_alphabet(data.test_dir_1, data.test_dir_2)
    data.LM_build_alphabet(data.LM_dir_1, data.LM_dir_2)
    data.build_task_domain_alphabet()
    data.fix_alphabet()

    for i in range(data.label_alphabet_1.size()-1):
        print(data.label_alphabet_1.instances[i])
    for i in range(data.label_alphabet_2.size()-1):
        print(data.label_alphabet_2.instances[i])


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
    batch_size = gold_variable.size(0)
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
        assert(len(pred)==len(gold))
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
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label



# def save_data_setting(data, save_file):
#     new_data = copy.deepcopy(data)
#     ## remove input instances
#     new_data.train_texts = []
#     new_data.dev_texts = []
#     new_data.test_texts = []
#     new_data.raw_texts = []

#     new_data.train_Ids = []
#     new_data.dev_Ids = []
#     new_data.test_Ids = []
#     new_data.raw_Ids = []
#     ## save data settings
#     with open(save_file, 'w') as fp:
#         pickle.dump(new_data, fp)
#     print("Data setting saved to file:",save_file)


# def load_data_setting(save_file):
#     with open(save_file, 'r') as fp:
#         data = pickle.load(fp)
#     print("Data setting loaded from file: ", save_file)
#     data.show_data_summary()
#     return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances_1 = data.train_Ids_1
        instances_2 = data.train_Ids_2
    elif name == "dev":
        instances_1 = data.dev_Ids_1
        instances_2 = data.dev_Ids_2
    elif name == 'test':
        instances_1 = data.test_Ids_1
        instances_2 = data.test_Ids_2
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()

    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    train_num = len(instances_1)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances_1[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, lm_seq_tensor, mask = batchify_with_label(instance, data.HP_gpu, True)

        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq_1 = model('model2', batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq_1, batch_label, mask, data.label_alphabet_1, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    # decode_time = time.time() - start_time
    # speed = len(instances)/decode_time
    acc_1, p_1, r_1, f_1 = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)

    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    train_num = len(instances_2)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances_2[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, lm_seq_tensor, mask = batchify_with_label(instance, data.HP_gpu, True)

        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq_2 = model('model4', batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq_2, batch_label, mask, data.label_alphabet_2, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    # decode_time = time.time() - start_time
    # speed = len(instances)/decode_time
    acc_2, p_2, r_2, f_2 = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    acc = [acc_1, acc_2]
    p = [p_1, p_2]
    r = [r_1, r_2]
    f= [f_1, f_2]
    decode_time = time.time() - start_time
    speed = (len(instances_1) + len(instances_2))/decode_time
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
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()

    lm_forward_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    lm_backward_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()

    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long())
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        #if seqlen<=1: continue
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        if seqlen > 1:
            lm_forward_seq_tensor[idx, 0: seqlen-1] = word_seq_tensor[idx, 1: seqlen]
            lm_forward_seq_tensor[idx, seqlen - 1] = torch.LongTensor([1]) # unk word
            lm_backward_seq_tensor[idx, 1: seqlen] = word_seq_tensor[idx, 0: seqlen-1]
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1]) # unk word
        else:
            lm_forward_seq_tensor[idx, 0] = torch.LongTensor([1]) # unk word
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])

    #lm_forward_seq_tensor[:, 0:-1] = word_seq_tensor[:, 1:]
    #lm_backward_seq_tensor[:, 1:] = word_seq_tensor[:, 0:-1]

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]

    lm_forward_seq_tensor = lm_forward_seq_tensor[word_perm_idx]
    lm_backward_seq_tensor = lm_backward_seq_tensor[word_perm_idx]

    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()

        lm_forward_seq_tensor = lm_forward_seq_tensor.cuda()
        lm_backward_seq_tensor = lm_backward_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    lm_seq_tensor = [lm_forward_seq_tensor, lm_backward_seq_tensor]
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor,lm_seq_tensor, mask


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    model = SeqModel(data)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    dev_f = []
    test_f = []
    perplexity_1 = []
    perplexity_2 = []
    best_epoch = 0
    # data.HP_iteration = 1
    LM_data = data.train_Ids_2
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
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
        random.shuffle(data.train_Ids_1)
        random.shuffle(data.train_Ids_2)

        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0

        ###co-train for 4 models
        train_num_1 = len(data.train_Ids_1)
        train_num_2 = len(data.train_Ids_2)
        train_num_3 = len(LM_data)
        total_batch_1 = train_num_1//batch_size+1
        batch_size_2 = train_num_2//total_batch_1
        l_batch_num_2 = train_num_2 - total_batch_1*batch_size_2

        start_2 = end_2 = 0

        for batch_id in range(total_batch_1):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            start_2 = end_2
            if batch_id < l_batch_num_2:
                end_2 = start_2 + (batch_size_2 + 1)
            else:
                end_2 = start_2 + batch_size_2

            if end >train_num_1:
                end = train_num_1
            if end_2 >train_num_2:
                end_2= train_num_2


            instance_1 = data.train_Ids_1[start:end]
            instance_2 = data.train_Ids_2[start_2:end_2]

            if not instance_1 or not instance_2:
                continue
            #seq label
            batch_word_1, batch_features_1, batch_wordlen_1, batch_wordrecover_1, batch_char_1, batch_charlen_1, batch_charrecover_1, batch_label_1, lm_seq_tensor_1, mask_1 = batchify_with_label(instance_1, data.HP_gpu)
            batch_word_2, batch_features_2, batch_wordlen_2, batch_wordrecover_2, batch_char_2, batch_charlen_2, batch_charrecover_2, batch_label_2, lm_seq_tensor_2, mask_2 = batchify_with_label(instance_2, data.HP_gpu)

            batch_word=[batch_word_1, batch_word_2]
            batch_features=[batch_features_1, batch_features_2]
            batch_wordlen=[batch_wordlen_1, batch_wordlen_2]
            batch_char=[batch_char_1, batch_char_2]
            batch_charlen=[batch_charlen_1, batch_charlen_2]
            batch_charrecover=[batch_charrecover_1, batch_charrecover_2]
            batch_label=[batch_label_1 ,batch_label_2]
            lm_seq_tensor=[lm_seq_tensor_1, lm_seq_tensor_2]
            mask=[mask_1, mask_2]
            instance_count += 1
            loss_ = []
            perplexity_ = []

            # LM 1
            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = model.loss('model1', batch_word[0], batch_features[0], batch_wordlen[0], batch_char[0], batch_charlen[0], batch_charrecover[0], batch_label[0], lm_seq_tensor[0], mask[0])

            loss_.append(loss)
            perplexity_.append(perplexity)

            #seq label 1
            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = model.loss('model2', batch_word[0], batch_features[0], batch_wordlen[0], batch_char[0], batch_charlen[0], batch_charrecover[0], batch_label[0], lm_seq_tensor[0], mask[0])
            loss_.append(loss)

            # LM 2
            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = model.loss('model3', batch_word[1], batch_features[1], batch_wordlen[1], batch_char[1], batch_charlen[1], batch_charrecover[1], batch_label[1], lm_seq_tensor[1], mask[1])

            loss_.append(loss)
            perplexity_.append(perplexity)

            #seq label 2
            loss, perplexity, tag_seq_forward, tag_seq_backward, tag_seq = model.loss('model4', batch_word[1], batch_features[1], batch_wordlen[1], batch_char[1], batch_charlen[1], batch_charrecover[1], batch_label[1], lm_seq_tensor[1], mask[1])
            loss_.append(loss)

            loss_rate = [1.0, 1.0, 1.0, 2.0]
            loss = 0
            model_num = len(loss_)
            for loss_id in range(model_num):
                loss += loss_[loss_id] * loss_rate[loss_id]
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total_loss_1 += loss_[0].data[0]
            total_loss_2 += loss_[1].data[0]
            total_loss_3 += loss_[2].data[0]
            total_loss_4 += loss_[3].data[0]
            total_perplexity_1 += perplexity_[0].data[0]
            total_perplexity_2 += perplexity_[1].data[0]

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        LM_perplex_1 = math.exp(total_perplexity_1/total_batch_1)
        LM_perplex_2 = math.exp(total_perplexity_2/total_batch_1)
        perplexity_1.append(LM_perplex_1)
        perplexity_2.append(LM_perplex_2)
        print("Epoch: %s training finished. Time: %.2fs" % (idx, epoch_cost))
        print("Epoch: %s training finished. Time: %.2fs,   total loss: %s"%(idx, epoch_cost,  total_loss_2))
        print("totalloss:", total_loss_2)
        print("Epoch: %s training finished. Time: %.2fs,  total perplexity: %.4f" % (idx, epoch_cost,  LM_perplex_1))
        print("Epoch: %s training finished. Time: %.2fs,   total loss: %s"%(idx, epoch_cost, total_loss_4))
        print("totalloss:", total_loss_4)
        print("Epoch: %s training finished. Time: %.2fs,   total perplexity: %.4f" % (idx, epoch_cost,  LM_perplex_2))

        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev")
        dev_f.append(f[1])
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f[1]
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc[0], p[0], r[0], f[0]))
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc[1], p[1], r[1], f[1]))
        else:
            current_score = acc[1]
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc[0]))
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc[1]))

        if current_score > best_dev:
            best_epoch = idx
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            # model_name = data.model_dir +'.'+ str(idx) + ".model"
            model_name = data.model_dir + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        # ## decode test
        speed, acc, p, r, f, _,_ = evaluate(data, model, "test")
        test_f.append(f[1])
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc[0], p[0], r[0], f[0]))
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc[1], p[1], r[1], f[1]))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc[0]))
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc[1]))
        gc.collect()

    print("the best dev score is in epoch %s, dev:%.4f, test:%.4f" % (best_epoch, dev_f[best_epoch], test_f[best_epoch]))
    with open('data/fscore_13PC.txt', 'w') as ft:
        ft.write('dev f scores:\n')
        for t in dev_f:
            ft.write(str(round(t, 6)))
            ft.write(' ')
        ft.write('\n')
        ft.write('test f scores:\n')
        for t in test_f:
            ft.write(str(round(t, 6)))
            ft.write(' ')
        ft.write('\n')
        ft.write('LM 1 perplexity:\n')
        for t in perplexity_1:
            ft.write(str(round(t, 6)))
            ft.write(' ')
        ft.write('\n')
        ft.write('LM 2 perplexity:\n')
        for t in perplexity_2:
            ft.write(str(round(t, 6)))
            ft.write(' ')

    if data.task_emb_save_dir is not None:
        with open('data/task_emb.txt', 'w') as ft:
            for task, i in data.task_alphabet.iteritems():
                ft.write(task)
                ft.write(' ')
                for t in model.word_hidden.LSTM_param_generator.task_emb_vocab.weight.data[i]:
                    ft.write(str(round(t, 6)))
                    ft.write(' ')
                ft.write('\n')
    if data.domain_emb_save_dir is not None:
        with open('data/domain_emb.txt', 'w') as fd:
            for domain, i in data.domain_alphabet.iteritems():
                fd.write(domain)
                fd.write(' ')
                for t in model.word_hidden.LSTM_param_generator.domain_emb_vocab.weight.data[i]:
                    fd.write(str(round(t, 6)))
                    fd.write(' ')
                fd.write('\n')




def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File')

    args = parser.parse_args()
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    #data.HP_gpu = False
    data.read_config(args.config)
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
        # data.save_task_domain_embedding()

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        print(data.raw_dir)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

