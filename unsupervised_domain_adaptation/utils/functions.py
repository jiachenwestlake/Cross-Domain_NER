# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-06-10 17:49:50
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


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length,
                  char_padding_size=-1, char_padding_symbol='</pad>'):

    instance_text = []
    instance_idx = []
    words = []
    chars = []
    labels = []

    word_idx = []
    char_idx = []
    label_idx = []

    in_lines = open(input_file, 'r').readlines()

    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]

            if number_normalized:
                word = normalize_word(word)

            words.append(word)
            labels.append(label)
            word_idx.append(word_alphabet.get_index(word))
            label_idx.append(label_alphabet.get_index(label))

            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                # not padding
                pass
            for char in char_list:
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_idx.append(char_id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                instance_text.append([words, chars, labels])
                instance_idx.append([word_idx, char_idx, label_idx])
            words = []
            chars = []
            labels = []

            word_idx = []
            char_idx = []
            label_idx = []
    return instance_text, instance_idx


def read_instance_lm(input_file, word_alphabet, char_alphabet, number_normalized, max_sent_length, char_padding_size=-1,
                     char_padding_symbol='</pad>'):

    instance_texts = []
    instance_idx = []

    words = []
    chars = []
    labels = []

    word_idx = []
    char_idx = []
    label_idx = []

    in_lines = open(input_file, 'r', encoding="utf-8").readlines()
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]

            if number_normalized:
                word = normalize_word(word)

            words.append(word)
            labels.append(label)

            word_idx.append(word_alphabet.get_index(word))
            label_idx.append(1)

            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_idx.append(char_id)
        else:
            if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                instance_texts.append([words, chars, labels])
                instance_idx.append([word_idx, char_idx, label_idx])

            words = []
            chars = []
            labels = []

            word_idx = []
            char_idx = []
            label_idx = []
    return instance_texts, instance_idx


def read_instance_lm_raw(input_file, word_alphabet, char_alphabet, number_normalized,
                         char_padding_size=-1, char_padding_symbol='</pad>'):

    instance_texts = []
    instance_idx = []

    in_lines = open(input_file, 'r', encoding="utf-8").readlines()

    for line in in_lines:

        words = []
        chars = []
        labels = []

        word_idx = []
        char_idx = []
        label_idx = []

        pairs = line.strip().split()
        for word in pairs:
            if number_normalized:
                word = normalize_word(word)

            label = '\0'
            words.append(word)
            labels.append(label)
            word_idx.append(word_alphabet.get_index(word))
            label_idx.append(1)

            ## get char
            char_list = []
            char_id = []
            for char in word:
                char_list.append(char)

            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_idx.append(char_id)

        if not len(words):
            continue
        instance_texts.append([words, chars, labels])
        instance_idx.append([word_idx, char_idx, label_idx])

    return instance_texts, instance_idx


def build_pretrain_embedding(embedding_path, word_alphabet, embed_dim=100, norm=False):
    embed_dict = dict()
    if embedding_path:
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
    pretrain_size = len(embed_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" %
          (pretrain_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embed_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embed_dim = -1
    embed_dict = dict()
    with open(embedding_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embed_dim < 0:
                embed_dim = len(tokens) - 1
            else:
                assert (embed_dim + 1 == len(tokens))
            embed = np.empty([1, embed_dim])
            embed[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embed_dict[first_col] = embed
    return embed_dict, embed_dim


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
