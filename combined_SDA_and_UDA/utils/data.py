# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Xiaobo Liang and Chen Jia
# @Last Modified time: 2019-05-20
from __future__ import print_function
from __future__ import absolute_import
from .alphabet import Alphabet
from .functions import *
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class Data:
    def __init__(self):

        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1

        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.char_alphabet = Alphabet('character')
        self.task_alphabet = Alphabet('task')
        self.domain_alphabet = Alphabet('domain')

        self.seg = True

        # supervised learning
        self.supervised_ner_1_train = None
        self.supervised_ner_1_dev = None
        self.supervised_ner_1_test = None

        self.supervised_ner_2_train = None
        self.supervised_ner_2_dev = None
        self.supervised_ner_2_test = None

        self.supervised_lm_1_train = None
        self.supervised_lm_2_train = None

        # transfer learning
        self.transfer_ner_1_train = None
        self.transfer_ner_1_dev = None
        self.transfer_ner_1_test = None

        self.transfer_ner_2_dev = None
        self.transfer_ner_2_test = None

        self.transfer_lm_1_train = None
        self.transfer_lm_2_train = None

        # check point
        self.init_dir = None
        self.model_dir = None

        self.word_emb_dir = None

        self.ner_1_train_text = []
        self.ner_1_dev_text = []
        self.ner_1_test_text = []

        self.ner_2_train_text = []
        self.ner_2_dev_text = []
        self.ner_2_test_text = []

        self.lm_1_text = []
        self.lm_2_text = []

        self.ner_1_train_idx = []
        self.ner_1_dev_idx = []
        self.ner_1_test_idx = []

        self.ner_2_train_idx = []
        self.ner_2_dev_idx = []
        self.ner_2_test_idx = []

        self.lm_1_idx = []
        self.lm_2_idx = []

        self.word_alphabet = Alphabet('word')

        self.label_alphabet_ner_1 = Alphabet('label', True)
        self.label_alphabet_ner_2 = Alphabet('label', True)

        self.pretrain_word_embedding = None

        self.word_alphabet_size = 0
        self.label_alphabet_ner_1_size = 0
        self.label_alphabet_ner_2_size = 0
        self.char_alphabet_size = 0
        self.task_alphabet_size = 0
        self.domain_alphabet_size = 0

        self.word_emb_dim = 100
        self.char_emb_dim = 50

        self.task_emb_dim = 8
        self.domain_emb_dim = 8

        self.task_number = 2
        self.domain_number = 2

        self.ner_task_name = 'ner'
        self.domain_1_name = 'domain_1'
        self.lm_task_name = 'lm'
        self.domain_2_name = 'domain_2'

        # Networks
        self.use_char = True
        self.char_feature_extractor = "CNN"
        self.use_ner_crf = True

        # Training
        self.average_batch_loss = False
        self.optimizer = "SGD"
        self.status = "train"
        self.mode = 'supervised'

        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_LM = False

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_cpg = 0.005
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        print("++" * 50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     MAX SENTENCE LENGTH: %s" % self.MAX_SENTENCE_LENGTH)
        print("     MAX   WORD   LENGTH: %s" % self.MAX_WORD_LENGTH)
        print("     Number   normalized: %s" % self.number_normalized)
        print("     Word  alphabet size: %s" % self.word_alphabet_size)
        print("     Char  alphabet size: %s" % self.char_alphabet_size)
        print("     Label 1 size: %s, Label 2 size: %s" % (self.label_alphabet_ner_1_size,
                                                           self.label_alphabet_ner_2_size))
        print("     Word embedding  dir: %s" % self.word_emb_dir)
        print("     Supervised train: ")
        print("     Train  file directory: %s, Train  file directory: %s" % (self.supervised_ner_1_train,
                                                                             self.supervised_ner_2_train))
        print("     Dev    file directory: %s, Dev    file directory: %s" % (self.supervised_ner_1_dev,
                                                                             self.supervised_ner_2_dev))
        print("     Test   file directory: %s, Test   file directory: %s" % (self.supervised_ner_1_test,
                                                                             self.supervised_ner_2_test))
        print("     Transfer train: ")
        print("     Train  file directory: %s" % self.transfer_ner_1_train)
        print("     Dev    file directory: %s, Dev    file directory: %s" % (self.transfer_ner_1_dev,
                                                                             self.transfer_ner_2_dev))
        print("     Test   file directory: %s, Test   file directory: %s" % (self.transfer_ner_1_test,
                                                                             self.transfer_ner_2_test))

        print("     init   file directory: %s" % self.init_dir)
        print("     Model  file directory: %s" % self.model_dir)
        print("     Train instance number: %s, Train instance number: %s" % (len(self.ner_1_train_text),
                                                                             len(self.ner_2_train_text)))
        print("     Dev   instance number: %s, Dev   instance number: %s" % (len(self.ner_1_dev_text),
                                                                             len(self.ner_2_dev_text)))
        print("     Test  instance number: %s, Test  instance number: %s" % (len(self.ner_1_test_text),
                                                                             len(self.ner_2_test_text)))

        print("     LM  instance number: %s, LM  instance number: %s" % (len(self.lm_1_text), len(self.lm_2_text)))
        print(" " + "++" * 20)
        print(" Model Network:")
        print("     Model  use_ner_crf: %s" % self.use_ner_crf)
        print("     Model       use_char: %s" % self.use_char)
        if self.use_char:
            print("     Model char extractor: %s" % self.char_feature_extractor)
            print("     Model char_hidden_dim: %s" % self.HP_char_hidden_dim)
        print(" " + "++" * 20)
        print(" Training:")
        print("     Optimizer: %s" % self.optimizer)
        print("     Iteration: %s" % self.HP_iteration)
        print("     BatchSize: %s" % self.HP_batch_size)
        print('     Average  batch   loss: %s' % self.average_batch_loss)

        print(" " + "++" * 20)
        print(" Hyperparameters:")
        print("     Hyper              lr: %s" % self.HP_lr)
        print("     Hyper        lr_decay: %s" % self.HP_lr_decay)
        print("     Hyper         HP_clip: %s" % self.HP_clip)
        print("     Hyper        momentum: %s" % self.HP_momentum)
        print("     Hyper              l2: %s" % self.HP_l2)
        print("     Hyper      hidden_dim: %s" % self.HP_hidden_dim)
        print("     Hyper         dropout: %s" % self.HP_dropout)
        print("     Hyper      lstm_layer: %s" % self.HP_lstm_layer)
        print("     Hyper          bilstm: %s" % self.HP_bilstm)
        print("     Hyper             GPU: %s" % self.HP_gpu)
        print("DATA SUMMARY END.")
        print("++" * 50)
        sys.stdout.flush()

    def filter_word_count(self):
        new_d1_vocab = Alphabet("filter_word")
        for word, index in self.word_alphabet.iteritems():
            if self.word_alphabet.get_count(word) > 2:
                new_d1_vocab.add(word)
        self.word_alphabet = new_d1_vocab
        self.word_alphabet_size = new_d1_vocab.size()
        print("new vocab size {}".format(self.word_alphabet_size))

    def build_alphabet(self, ner_1_file, ner_2_file, single_label_alphabet=False):
        if ner_1_file is not None:
            for line in open(ner_1_file, 'r').readlines():
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    if self.number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]
                    self.label_alphabet_ner_1.add(label)
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)

        if ner_2_file is not None:
            for line in open(ner_2_file, 'r').readlines():
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if sys.version_info[0] < 3:
                        word = word.decode('utf-8')
                    if self.number_normalized:
                        word = normalize_word(word)
                    label = pairs[-1]
                    if single_label_alphabet:
                        self.label_alphabet_ner_1.add(label)
                    else:
                        self.label_alphabet_ner_2.add(label)
                    self.word_alphabet.add(word)
                    for char in word:
                        self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_ner_1_size = self.label_alphabet_ner_1.size()
        self.label_alphabet_ner_2_size = self.label_alphabet_ner_2.size()

    def build_language_model_alphabet(self, lm_1_file=None, lm_2_file=None):
        for line in open(lm_1_file).readlines():
            pairs = line.strip().split()
            for word in pairs:
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

        for line in open(lm_2_file).readlines():
            pairs = line.strip().split()
            for word in pairs:
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

        self.char_alphabet_size = self.char_alphabet.size()
        self.word_alphabet_size = self.word_alphabet.size()

    def build_task_domain_alphabet(self):

        self.task_alphabet.add("ner")
        self.task_alphabet.add("lm")
        self.task_alphabet_size = self.task_alphabet.size()

        self.domain_alphabet.add("domain_1")
        self.domain_alphabet.add("domain_2")
        self.domain_alphabet_size = self.domain_alphabet.size()

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet_ner_1.close()
        self.label_alphabet_ner_2.close()
        self.task_alphabet.close()
        self.domain_alphabet.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pre-trained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim =\
                build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)

    def generate_instance(self, instance_type, transfer_flag=False):
        self.fix_alphabet()
        if instance_type == "train":
            if transfer_flag:
                self.ner_1_train_text, self.ner_1_train_idx = \
                    read_ner_instance(self.transfer_ner_1_train, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)

                self.lm_1_text, self.lm_1_idx =\
                    read_lm_instance(self.transfer_lm_1_train, self.word_alphabet, self.char_alphabet,
                                     self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.lm_2_text, self.lm_2_idx =\
                    read_lm_instance(self.transfer_lm_2_train, self.word_alphabet, self.char_alphabet,
                                     self.number_normalized, self.MAX_SENTENCE_LENGTH)
            else:
                self.ner_1_train_text, self.ner_1_train_idx = \
                    read_ner_instance(self.supervised_ner_1_train, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.ner_2_train_text, self.ner_2_train_idx = \
                    read_ner_instance(self.supervised_ner_2_train, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)

                self.lm_1_text, self.lm_1_idx =\
                    read_lm_instance(self.supervised_lm_1_train, self.word_alphabet, self.char_alphabet,
                                     self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.lm_2_text, self.lm_2_idx =\
                    read_lm_instance(self.supervised_lm_2_train, self.word_alphabet, self.char_alphabet,
                                     self.number_normalized, self.MAX_SENTENCE_LENGTH)

        elif instance_type == "dev":

            if not transfer_flag:
                self.ner_1_dev_text, self.ner_1_dev_idx = \
                    read_ner_instance(self.supervised_ner_1_dev, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.ner_2_dev_text, self.ner_2_dev_idx = \
                    read_ner_instance(self.supervised_ner_2_dev, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            else:
                self.ner_1_dev_text, self.ner_1_dev_idx = \
                    read_ner_instance(self.transfer_ner_1_dev, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.ner_2_dev_text, self.ner_2_dev_idx = \
                    read_ner_instance(self.transfer_ner_2_dev, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)

        elif instance_type == "test":

            if not transfer_flag:

                self.ner_1_test_text, self.ner_1_test_idx = \
                    read_ner_instance(self.supervised_ner_1_test, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.ner_2_test_text, self.ner_2_test_idx = \
                    read_ner_instance(self.supervised_ner_2_test, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            else:
                self.ner_1_test_text, self.ner_1_test_idx = \
                    read_ner_instance(self.transfer_ner_1_test, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
                self.ner_2_test_text, self.ner_2_test_idx = \
                    read_ner_instance(self.transfer_ner_2_test, self.word_alphabet, self.char_alphabet,
                                      self.label_alphabet_ner_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % instance_type)

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def read_config(self, config_file):
        config = config_file_to_dict(config_file)

        the_item = 'supervised_ner_1_train'
        if the_item in config:
            self.supervised_ner_1_train = config[the_item]

        the_item = 'supervised_ner_1_dev'
        if the_item in config:
            self.supervised_ner_1_dev = config[the_item]

        the_item = 'supervised_ner_1_test'
        if the_item in config:
            self.supervised_ner_1_test = config[the_item]

        the_item = 'supervised_ner_2_train'
        if the_item in config:
            self.supervised_ner_2_train = config[the_item]

        the_item = 'supervised_ner_2_dev'
        if the_item in config:
            self.supervised_ner_2_dev = config[the_item]

        the_item = 'supervised_ner_2_test'
        if the_item in config:
            self.supervised_ner_2_test = config[the_item]

        the_item = 'supervised_lm_1_train'
        if the_item in config:
            self.supervised_lm_1_train = config[the_item]

        the_item = 'supervised_lm_2_train'
        if the_item in config:
            self.supervised_lm_2_train = config[the_item]

        the_item = 'transfer_ner_1_train'
        if the_item in config:
            self.transfer_ner_1_train = config[the_item]

        the_item = 'transfer_ner_1_dev'
        if the_item in config:
            self.transfer_ner_1_dev = config[the_item]

        the_item = 'transfer_ner_1_test'
        if the_item in config:
            self.transfer_ner_1_test = config[the_item]

        the_item = 'transfer_ner_2_dev'
        if the_item in config:
            self.transfer_ner_2_dev = config[the_item]

        the_item = 'transfer_ner_2_test'
        if the_item in config:
            self.transfer_ner_2_test = config[the_item]

        the_item = 'transfer_lm_1_train'
        if the_item in config:
            self.transfer_lm_1_train = config[the_item]

        the_item = 'transfer_lm_2_train'
        if the_item in config:
            self.transfer_lm_2_train = config[the_item]

        the_item = 'init_dir'
        if the_item in config:
            self.init_dir = config[the_item]

        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]

        the_item = 'word_embed_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]

        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])

        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])

        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])

        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])

        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])

        the_item = 'task_emb_dim'
        if the_item in config:
            self.task_emb_dim = int(config[the_item])

        the_item = 'domain_emb_dim'
        if the_item in config:
            self.domain_emb_dim = int(config[the_item])

        # read network:
        the_item = 'use_ner_crf'
        if the_item in config:
            self.use_ner_crf = str2bool(config[the_item])

        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])

        # read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]

        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])

        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        the_item = 'mode'
        if the_item in config:
            self.mode = config[the_item]

        the_item = 'cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])

        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])

        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])

        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])

        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])

        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])

        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])

        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])

        the_item = 'learning_rate_cpg'
        if the_item in config:
            self.HP_lr_cpg = float(config[the_item])

        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])

        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])

        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])

        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item in config:
                print("Warning: duplicated config item found: %s, updated." % (pair[0]))
            config[item] = pair[-1]
    return config


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False
