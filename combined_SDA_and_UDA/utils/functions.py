# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_ner_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length):
    instance_text = []
    instance_idx = []
    words = []
    chars = []
    labels = []

    word_idx = []
    char_idx = []
    label_idx = []

    for line in open(input_file, 'r').readlines():
        if len(line) > 2:
            pairs = line.strip().split()

            if sys.version_info[0] < 3:
                word = pairs[0].decode('utf-8')
            else:
                word = pairs[0]

            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_idx.append(word_alphabet.get_index(word))
            label_idx.append(label_alphabet.get_index(label))

            # get char
            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_idx.append(char_id)
        else:
            if len(words) > 0 and len(words) < max_sent_length:
                instance_text.append([words, chars, labels])
                instance_idx.append([word_idx, char_idx, label_idx])
            words = []
            chars = []
            labels = []

            word_idx = []
            char_idx = []
            label_idx = []

    return instance_text, instance_idx


def read_lm_instance(input_file, word_alphabet, char_alphabet, number_normalized, max_sent_length):
    instance_text = []
    instance_idx = []
    words = []
    chars = []
    labels = []

    word_idx = []
    char_idx = []
    label_idx = []

    for line in open(input_file, 'r').readlines():
        pairs = line.strip().split()
        if len(pairs) < 1:
            continue
        for word in pairs:
            if sys.version_info[0] < 3:
                word = word.decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = '\0'
            words.append(word)
            labels.append(label)
            word_idx.append(word_alphabet.get_index(word))
            label_idx.append(1)

            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_idx.append(char_id)
        if len(words) > 0 and len(words) < max_sent_length:
            instance_text.append([words, chars, labels])
            instance_idx.append([word_idx, char_idx, label_idx])

        words = []
        chars = []
        labels = []

        word_idx = []
        char_idx = []
        label_idx = []

    return instance_text, instance_idx


def build_pretrain_embedding(embedding_path, word_alphabet, embed_dim=100, norm=True):
    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embed_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index, :] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embed_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        len(embed_dict), perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embed_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim
