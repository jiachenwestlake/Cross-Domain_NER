# -*- coding: utf-8 -*-
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

        # word or char vocab
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.source_label_alphabet = Alphabet('label', True)
        self.target_label_alphabet = Alphabet('label', True)

        # task1 task2 domain1 domain2
        self.task_alphabet = Alphabet('task')
        self.domain_alphabet = Alphabet('domain')

        # I/O
        self.source_train_dir = None
        self.source_dev_dir = None
        self.source_test_dir = None

        self.target_train_dir = None
        self.target_dev_dir = None
        self.target_test_dir = None

        self.source_lm_dir = None
        self.target_lm_dir = None

        self.decode_dir = None
        self.dset_dir = None  ## data vocabulary related file
        self.model_dir = None  ## model save  file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = None

        self.source_train_texts = []
        self.source_dev_texts = []
        self.source_test_texts = []
        self.target_train_texts = []
        self.target_dev_texts = []
        self.target_test_texts = []

        self.source_lm_texts = []
        self.target_lm_texts = []

        self.source_train_idx = []
        self.source_dev_idx = []
        self.source_test_idx = []

        self.target_train_idx = []
        self.target_dev_idx = []
        self.target_test_idx = []

        self.source_lm_idx = []
        self.target_lm_idx = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_task_embedding = None
        self.pretrain_domain_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.source_lm_word_alphabet_size = 0
        self.target_lm_word_alphabet_size = 0
        self.source_label_alphabet_size = 0
        self.target_label_alphabet_size = 0

        self.task_alphabet_size = 0
        self.domain_alphabet_size = 0

        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.task_emb_dim = 50
        self.domain_emb_dim = 50
        self.task_number = 2
        self.domain_number = 2
        self.LM_sample_num = 50

        self.model1_task_name = 'ner_task'
        self.model1_domain_name = 'source_domain'
        self.model2_task_name = 'lm_task'
        self.model2_domain_name = 'source_domain'
        self.model3_task_name = 'ner_task'
        self.model3_domain_name = 'target_domain'
        self.model4_task_name = 'lm_task'
        self.model4_domain_name = 'target_domain'

        # Networks
        self.word_feature_extractor = "LSTM"
        self.use_char = True
        self.char_feature_extractor = "CNN"
        self.use_crf_sl = True
        self.use_crf_lm = False
        self.nbest = None

        # Training
        self.average_batch_loss = False
        self.optimizer = "SGD"
        self.status = "train"

        # Hyperparameters
        self.HP_cnn_layer = 4
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_LM = False
        self.adv_train_flag = True

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
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Label alphabet size: %s, Label alphabet size: %s" % (self.source_label_alphabet_size,
                                                                         self.target_label_alphabet_size))
        print("     Word embedding  dir: %s" % (self.word_emb_dir))

        print("     Train  file directory: %s, Train  file directory: %s" % (self.source_train_dir,
                                                                             self.target_train_dir))
        print("     Dev    file directory: %s, Dev    file directory: %s" % (self.source_dev_dir,
                                                                             self.target_dev_dir))
        print("     Test   file directory: %s, Test   file directory: %s" % (self.source_test_dir,
                                                                             self.target_test_dir))
        print("     Dset   file directory: %s" % (self.dset_dir))
        print("     Model  file directory: %s" % (self.model_dir))
        print("     Loadmodel   directory: %s" % (self.load_model_dir))
        print("     Decode file directory: %s" % (self.decode_dir))
        print("     Train instance number: %s, Train instance number: %s" % (len(self.source_train_texts),
                                                                             len(self.target_train_texts)))
        print("     Dev   instance number: %s, Dev   instance number: %s" % (len(self.source_dev_texts),
                                                                             len(self.target_dev_texts)))
        print("     Test  instance number: %s, Test  instance number: %s" % (len(self.source_test_texts),
                                                                             len(self.target_test_texts)))

        print("     LM  instance number: %s, LM  instance number: %s" % (len(self.source_lm_texts),
                                                                         len(self.target_lm_texts)))

        print(" " + "++" * 20)

        print(" Model Network:")
        print("     Model    sl    use_crf: %s, Model    lm    use_crf: %s" % (self.use_crf_sl, self.use_crf_lm))
        print("     Model word extractor: %s" % (self.word_feature_extractor))
        print("     Model       use_char: %s" % (self.use_char))
        if self.use_char:
            print("     Model char extractor: %s" % (self.char_feature_extractor))
            print("     Model char_hidden_dim: %s" % (self.HP_char_hidden_dim))
        print(" " + "++" * 20)
        print(" Training:")
        print("     Optimizer: %s" % (self.optimizer))
        print("     Iteration: %s" % (self.HP_iteration))
        print("     BatchSize: %s" % (self.HP_batch_size))
        print("     Average  batch   loss: %s" % (self.average_batch_loss))

        print(" " + "++" * 20)
        print(" Hyperparameters:")

        print("     Hyper              lr: %s" % (self.HP_lr))
        print("     Hyper        lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyper         HP_clip: %s" % (self.HP_clip))
        print("     Hyper        momentum: %s" % (self.HP_momentum))
        print("     Hyper              l2: %s" % (self.HP_l2))
        print("     Hyper      hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyper         dropout: %s" % (self.HP_dropout))
        print("     Hyper      lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyper          bilstm: %s" % (self.HP_bilstm))
        print("     Hyper             GPU: %s" % (self.HP_gpu))
        print("DATA SUMMARY END.")
        print("++" * 50)
        sys.stdout.flush()

    def build_alphabet(self, input_file, domain):
        lines = open(input_file, 'r', encoding="utf-8").readlines()
        for line in lines:
            if line.strip():
                pairs = line.strip().split()
                word = pairs[0]
                label = pairs[-1]

                if self.number_normalized:
                    word = normalize_word(word)

                if domain == "source":
                    self.source_label_alphabet.add(label)
                else:
                    self.target_label_alphabet.add(label)

                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.source_label_alphabet_size = self.source_label_alphabet.size()
        self.target_label_alphabet_size = self.target_label_alphabet.size()
        self.task_alphabet_size = self.task_alphabet.size()
        self.domain_alphabet_size = self.domain_alphabet.size()

    def build_alphabet_lm(self, input_file):
        if input_file is not None:
            in_lines = open(input_file, 'r', encoding="utf-8").readlines()
        else:
            in_lines = []

        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]

                if self.number_normalized:
                    word = normalize_word(word)

                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()

    def build_alphabet_lm_raw(self, input_file):
        if input_file is not None:
            in_lines = open(input_file, 'r', encoding="utf-8").readlines()
        else:
            in_lines = []

        for line in in_lines:
            pairs = line.strip().split()
            for word in pairs:

                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                for char in word:
                    self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()

    def build_task_domain_alphabet(self):
        assert self.task_number == 2
        self.task_alphabet.add("ner_task")
        self.task_alphabet.add("lm_task")
        self.task_alphabet_size = self.task_alphabet.size()
        assert self.domain_number == 2
        self.domain_alphabet.add("source_domain")
        self.domain_alphabet.add("target_domain")
        self.domain_alphabet_size = self.domain_alphabet.size()

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.source_label_alphabet.close()
        self.target_label_alphabet.close()
        self.task_alphabet.close()
        self.domain_alphabet.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, dir: %s" % (self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim)

    def filter_word_count(self):
        new_vocab = Alphabet("new_word")
        for word, index in self.word_alphabet.iteritems():
            if self.word_alphabet.get_count(word) > 3:
                new_vocab.add(word)
        self.word_alphabet = new_vocab
        self.word_alphabet_size = new_vocab.size()
        print("new vocab size {}".format(self.word_alphabet_size))

    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.source_train_texts, self.source_train_idx = \
                read_instance(self.source_train_dir, self.word_alphabet, self.char_alphabet, self.source_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)

            self.target_train_texts, self.target_train_idx = \
                read_instance(self.target_train_dir, self.word_alphabet, self.char_alphabet, self.target_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)

            self.source_lm_texts, self.source_lm_idx = \
                read_instance_lm_raw(self.source_lm_dir, self.word_alphabet, self.char_alphabet, self.number_normalized)

            self.target_lm_texts, self.target_lm_idx = \
                read_instance_lm_raw(self.target_lm_dir, self.word_alphabet, self.char_alphabet, self.number_normalized)

        elif name == "dev":
            self.source_dev_texts, self.source_dev_idx = \
                read_instance(self.source_dev_dir, self.word_alphabet, self.char_alphabet, self.source_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.target_dev_texts, self.target_dev_idx = \
                read_instance(self.target_dev_dir, self.word_alphabet, self.char_alphabet, self.target_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.source_test_texts, self.source_test_idx = \
                read_instance(self.source_test_dir, self.word_alphabet, self.char_alphabet, self.source_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.target_test_texts, self.target_test_idx = \
                read_instance(self.target_test_dir, self.word_alphabet, self.char_alphabet, self.target_label_alphabet,
                              self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, predict_results, name):
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts_2
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, self.decode_dir))

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts_2
        elif name == 'test':
            content_list = self.target_test_texts
        elif name == 'dev':
            content_list = self.target_dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        assert (sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f') + " "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy] + " "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s" % (name, nbest, self.decode_dir))

    def read_config(self, config_file):
        config = config_file_to_dict(config_file)

        # read data:
        the_item = 'source_train_dir'
        if the_item in config:
            self.source_train_dir = config[the_item]

        the_item = 'source_dev_dir'
        if the_item in config:
            self.source_dev_dir = config[the_item]

        the_item = 'source_test_dir'
        if the_item in config:
            self.source_test_dir = config[the_item]

        the_item = 'target_train_dir'
        if the_item in config:
            self.target_train_dir = config[the_item]

        the_item = 'target_test_dir_1'
        if the_item in config:
            self.target_dev_dir = config[the_item]

        the_item = 'target_test_dir_2'
        if the_item in config:
            self.target_test_dir = config[the_item]

        the_item = 'source_lm_dir'
        if the_item in config:
            self.source_lm_dir = config[the_item]

        the_item = 'target_lm_dir'
        if the_item in config:
            self.target_lm_dir = config[the_item]

        the_item = 'decode_dir'
        if the_item in config:
            self.decode_dir = config[the_item]
        the_item = 'dset_dir'
        if the_item in config:
            self.dset_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]

        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]

        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])
        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])

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
        the_item = 'sample_num'
        if the_item in config:
            self.LM_sample_num = int(config[the_item])

        # read network:
        the_item = 'use_crf_sl'
        if the_item in config:
            self.use_crf_sl = str2bool(config[the_item])
        the_item = 'use_crf_lm'
        if the_item in config:
            self.use_crf_lm = str2bool(config[the_item])

        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])
        the_item = 'word_seq_feature'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_seq_feature'
        if the_item in config:
            self.char_feature_extractor = config[the_item]
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])

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

        # read Hyperparameters:
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
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False
