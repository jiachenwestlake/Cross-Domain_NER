# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-06-22 00:01:47
from __future__ import print_function
from __future__ import absolute_import
import sys
from .alphabet import Alphabet
from .functions import *
import torch
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
        self.norm_task_emb = False
        self.norm_domain_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')
        self.task_alphabet = Alphabet('task')
        self.domain_alphabet = Alphabet('domain')

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None


        self.label_alphabet_1 = Alphabet('label',True)
        self.label_alphabet_2 = Alphabet('label',True)
        self.tagScheme = "NoSeg" ## BMES/BIO

        self.seg = True

        ### I/O
        self.train_dir = None
        self.train_dir_1 = None
        self.dev_dir_1 = None
        self.test_dir_1 = None
        self.train_dir_2 = None
        self.dev_dir_2 = None
        self.test_dir_2 = None
        self.LM_dir_1 = None
        self.LM_dir_2 = None
        self.raw_dir_1 = None
        self.raw_dir_2 = None

        self.decode_dir = None
        self.dset_dir = None ## data vocabulary related file
        self.model_dir = None ## model save  file
        self.load_model_dir = None ## model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.task_emb_dir = None
        self.domain_emb_dir = None
        self.task_emb_save_dir = None
        self.domain_emb_save_dir = None
        self.feature_emb_dirs = []

        self.train_texts_1 = []
        self.dev_texts_1 = []
        self.test_texts_1 = []
        self.train_texts_2 = []
        self.dev_texts_2 = []
        self.test_texts_2 = []
        self.LM_texts_1 = []
        self.LM_texts_2 = []
        self.raw_texts_1 = []
        self.raw_texts_2 = []

        self.train_Ids_1 = []
        self.dev_Ids_1 = []
        self.test_Ids_1 = []
        self.train_Ids_2 = []
        self.dev_Ids_2 = []
        self.test_Ids_2 = []
        self.LM_Ids_1 = []
        self.LM_Ids_2 = []
        self.raw_Ids_1 = []
        self.raw_Ids_2 = []
        # domain alphabet in LM prediction
        self.D1_word_alphabet = Alphabet('word')
        self.D2_word_alphabet = Alphabet('word')
        self.D1_word_alphabet_size = 0
        self.D2_word_alphabet_size = 0

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_task_embedding = None
        self.pretrain_domain_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_1_size = 0
        self.label_alphabet_2_size = 0
        self.task_alphabet_size = 0
        self.domain_alphabet_size = 0
        self.feature_alphabet_sizes = []
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30
        self.task_emb_dim = 50
        self.domain_emb_dim = 50
        self.task_number = 2
        self.domain_number = 2
        self.LM_sample_num = 50
        self.model1_task_name = 'task1'
        self.model1_domain_name = 'domain1'
        self.model2_task_name = 'task2'
        self.model2_domain_name = 'domain1'
        self.model3_task_name = 'task1'
        self.model3_domain_name = 'domain2'
        self.model4_task_name = 'task2'
        self.model4_domain_name = 'domain2'

        ###Networks
        self.word_feature_extractor = "LSTM" ## "LSTM"/"CNN"/"GRU"/
        self.use_char = True
        self.char_feature_extractor = "CNN" ## "LSTM"/"CNN"/"GRU"/None
        self.use_crf_sl = True
        self.use_crf_lm = False
        self.nbest = None
        self.LM_use_sample = True

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD" ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ### Hyperparameters
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
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        print("++"*50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     Tag          scheme: %s"%(self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s"%(self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s, D1 Word  alphabet size: %s, D2 Word  alphabet size: %s"%(self.word_alphabet_size, self.D1_word_alphabet_size, self.D2_word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s, Label alphabet size: %s"%(self.label_alphabet_1_size, self.label_alphabet_2_size))
        print("     Word embedding  dir: %s"%(self.word_emb_dir))
        print("     Char embedding  dir: %s"%(self.char_emb_dir))
        print("     Word embedding size: %s"%(self.word_emb_dim))
        print("     Char embedding size: %s"%(self.char_emb_dim))
        print("     Norm   word     emb: %s"%(self.norm_word_emb))
        print("     Norm   char     emb: %s"%(self.norm_char_emb))
        print("     Train  file directory: %s, Train  file directory: %s"%(self.train_dir_1, self.train_dir_2))
        print("     Dev    file directory: %s, Dev    file directory: %s"%(self.dev_dir_1, self.dev_dir_2))
        print("     Test   file directory: %s, Test   file directory: %s"%(self.test_dir_1, self.test_dir_2))
        print("     Raw  1  file directory: %s, Raw  2  file directory: %s"%(self.raw_dir_1, self.raw_dir_2))
        print("     Dset   file directory: %s"%(self.dset_dir))
        print("     Model  file directory: %s"%(self.model_dir))
        print("     Loadmodel   directory: %s"%(self.load_model_dir))
        print("     Decode file directory: %s"%(self.decode_dir))
        print("     Train instance number: %s, Train instance number: %s"%(len(self.train_texts_1), len(self.train_texts_2)))
        print("     Dev   instance number: %s, Dev   instance number: %s"%(len(self.dev_texts_1), len(self.dev_texts_2)))
        print("     Test  instance number: %s, Test  instance number: %s"%(len(self.test_texts_1), len(self.test_texts_2)))
        print("     LM  instance number: %s, LM  instance number: %s"%(len(self.LM_texts_1), len(self.LM_texts_2)))
        print("     Raw 1 instance number: %s, Raw 2 instance number: %s"%(len(self.raw_texts_1), len(self.raw_texts_2)))
        print("     FEATURE num: %s"%(self.feature_num))
        for idx in range(self.feature_num):
            print("         Fe: %s  alphabet  size: %s"%(self.feature_alphabets[idx].name, self.feature_alphabet_sizes[idx]))
            print("         Fe: %s  embedding  dir: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dirs[idx]))
            print("         Fe: %s  embedding size: %s"%(self.feature_alphabets[idx].name, self.feature_emb_dims[idx]))
            print("         Fe: %s  norm       emb: %s"%(self.feature_alphabets[idx].name, self.norm_feature_embs[idx]))
        print(" "+"++"*20)
        print(" Model Network:")
        print("     Model    sl    use_crf: %s, Model    lm    use_crf: %s"%(self.use_crf_sl, self.use_crf_lm))
        print("     Model word extractor: %s"%(self.word_feature_extractor))
        print("     Model       use_char: %s"%(self.use_char))
        if self.use_char:
            print("     Model char extractor: %s"%(self.char_feature_extractor))
            print("     Model char_hidden_dim: %s"%(self.HP_char_hidden_dim))
        print(" "+"++"*20)
        print(" Training:")
        print("     Optimizer: %s"%(self.optimizer))
        print("     Iteration: %s"%(self.HP_iteration))
        print("     BatchSize: %s"%(self.HP_batch_size))
        print("     Average  batch   loss: %s"%(self.average_batch_loss))

        print(" "+"++"*20)
        print(" Hyperparameters:")

        print("     Hyper              lr: %s"%(self.HP_lr))
        print("     Hyper        lr_decay: %s"%(self.HP_lr_decay))
        print("     Hyper         HP_clip: %s"%(self.HP_clip))
        print("     Hyper        momentum: %s"%(self.HP_momentum))
        print("     Hyper              l2: %s"%(self.HP_l2))
        print("     Hyper      hidden_dim: %s"%(self.HP_hidden_dim))
        print("     Hyper         dropout: %s"%(self.HP_dropout))
        print("     Hyper      lstm_layer: %s"%(self.HP_lstm_layer))
        print("     Hyper          bilstm: %s"%(self.HP_bilstm))
        print("     Hyper             GPU: %s"%(self.HP_gpu))
        print("DATA SUMMARY END.")
        print("++"*50)
        sys.stdout.flush()


    def initial_feature_alphabets(self):
        items = open(self.train_dir,'r').readline().strip('\n').split()
        total_column = len(items)
        if total_column > 2:
            for idx in range(1, total_column-1):
                feature_prefix = items[idx].split(']',1)[0]+"]"
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print("Find feature: ", feature_prefix)
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None]*self.feature_num
        self.feature_emb_dims = [20]*self.feature_num
        self.feature_emb_dirs = [None]*self.feature_num
        self.norm_feature_embs = [False]*self.feature_num
        self.feature_alphabet_sizes = [0]*self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[self.feature_name[idx]]['emb_norm']
        # exit(0)


    def build_alphabet(self, input_file_1, input_file_2):
        in_lines_1 = open(input_file_1,'r').readlines()
        for line in in_lines_1:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet_1.add(label)
                self.word_alphabet.add(word)
                self.D1_word_alphabet.add(word)  # domain 1 word alphabet
                ## build feature alphabet
                for idx in range(self.feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    self.feature_alphabets[idx].add(feat_idx)
                for char in word:
                    self.char_alphabet.add(char)
        in_lines_2 = open(input_file_2, 'r').readlines()
        for line in in_lines_2:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet_2.add(label)
                self.word_alphabet.add(word)
                self.D2_word_alphabet.add(word)  # domain 2 word alphabet
                ## build feature alphabet
                for idx in range(self.feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    self.feature_alphabets[idx].add(feat_idx)
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.D1_word_alphabet_size = self.D1_word_alphabet.size()
        self.D2_word_alphabet_size = self.D2_word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_1_size = self.label_alphabet_1.size()
        self.label_alphabet_2_size = self.label_alphabet_2.size()
        self.task_alphabet_size = self.task_alphabet.size()
        self.domain_alphabet_size = self.domain_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[idx] = self.feature_alphabets[idx].size()
        startS_1 = False
        startB_1 = False
        startS_2 = False
        startB_2 = False
        for label,_ in self.label_alphabet_1.iteritems():
            if "S-" in label.upper():
                startS_1 = True
            elif "B-" in label.upper():
                startB_1 = True
        for label,_ in self.label_alphabet_2.iteritems():
            if "S-" in label.upper():
                startS_2 = True
            elif "B-" in label.upper():
                startB_2 = True

        if startB_1 and startB_2:
            if startS_1 and startS_2:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
    # train data
    def build_alphabet_LM(self, input_file_1 = None, input_file_2 = None):
        if input_file_1 is not None:
            in_lines_1 = open(input_file_1,'r').readlines()
        else:
            in_lines_1 = []
        if input_file_2 is not None:
            in_lines_2 = open(input_file_2, 'r').readlines()
        else:
            in_lines_2 = []
        for line in in_lines_1:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                self.D1_word_alphabet.add(word)  # domain 1 word alphabet
                for char in word:
                    self.char_alphabet.add(char)
        for line in in_lines_2:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                self.D2_word_alphabet.add(word)  # domain 2 word alphabet
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.D1_word_alphabet_size = self.D1_word_alphabet.size()
        self.D2_word_alphabet_size = self.D2_word_alphabet.size()
    # raw data
    def LM_build_alphabet(self, input_file_1=None, input_file_2=None):
        if input_file_1 is not None:
            in_lines_1 = open(input_file_1, 'r').readlines()
        else:
            in_lines_1 = []
        if input_file_2 is not None:
            in_lines_2 = open(input_file_2, 'r').readlines()
        else:
            in_lines_2 = []
        for line in in_lines_1:
            pairs = line.strip().split()
            for word in pairs:
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                self.D1_word_alphabet.add(word) #domain 1 word alphabet
                for char in word:
                    self.char_alphabet.add(char)
        for line in in_lines_2:
            pairs = line.strip().split()
            for word in pairs:
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
                self.D2_word_alphabet.add(word)  # domain 2 word alphabet
                for char in word:
                    self.char_alphabet.add(char)

        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.D1_word_alphabet_size = self.D1_word_alphabet.size()
        self.D2_word_alphabet_size = self.D2_word_alphabet.size()

    def build_task_domain_alphabet(self):
        for i in range(self.task_number):
            task = 'task' + str(i+1)
            self.task_alphabet.add(task)
        self.task_alphabet_size = self.task_alphabet.size()
        for i in range(self.domain_number):
            domain = 'domain' +str(i+1)
            self.domain_alphabet.add(domain)
        self.domain_alphabet_size = self.domain_alphabet.size()


    def fix_alphabet(self):
        self.word_alphabet.close()
        self.D1_word_alphabet.close()
        self.D2_word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet_1.close()
        self.label_alphabet_2.close()
        self.task_alphabet.close()
        self.domain_alphabet.close()
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()


    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm: %s, dir: %s"%(self.norm_word_emb, self.word_emb_dir))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm: %s, dir: %s"%(self.norm_char_emb, self.char_emb_dir))
            self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)
        if self.task_emb_dir:
            print("Load pretrained task embedding, norm: %s, dir: %s"%(self.norm_task_emb, self.task_emb_dir))
            self.pretrain_task_embedding, self.task_emb_dim = build_pretrain_embedding(self.task_emb_dir, self.task_alphabet, self.task_emb_dim, self.norm_task_emb)
        if self.domain_emb_dir:
            print("Load pretrained task embedding, norm: %s, dir: %s"%(self.norm_domain_emb, self.domain_emb_dir))
            self.pretrain_domain_embedding, self.domain_emb_dim = build_pretrain_embedding(self.domain_emb_dir, self.domain_alphabet, self.domain_emb_dim, self.norm_domain_emb)
        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature %s embedding:, norm: %s, dir: %s"%(self.feature_name[idx], self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self.pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dims[idx], self.norm_feature_embs[idx])


    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts_1, self.train_Ids_1 = read_instance(self.train_dir_1, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.train_texts_2, self.train_Ids_2 = read_instance(self.train_dir_2, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts_1, self.dev_Ids_1 = read_instance(self.dev_dir_1, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.dev_texts_2, self.dev_Ids_2 = read_instance(self.dev_dir_2, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts_1, self.test_Ids_1 = read_instance(self.test_dir_1, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.test_texts_2, self.test_Ids_2 = read_instance(self.test_dir_2, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts_1, self.raw_Ids_1 = read_instance(self.raw_dir_1, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_1, self.number_normalized, self.MAX_SENTENCE_LENGTH)
            self.raw_texts_2, self.raw_Ids_2 = read_instance(self.raw_dir_2, self.word_alphabet, self.char_alphabet, self.feature_alphabets, self.label_alphabet_2, self.number_normalized, self.MAX_SENTENCE_LENGTH)

        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s"%(name))


    def write_decoded_results(self, predict_results, name):
        fout = open(self.decode_dir,'w')
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
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy].encode('utf-8') + " " + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s"%(name, self.decode_dir))


    def load(self,data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self,save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()


    def write_nbest_decoded_results(self, predict_results, pred_scores, name):
        ## predict_results : [whole_sent_num, nbest, each_sent_length]
        ## pred_scores: [whole_sent_num, nbest]
        fout = open(self.decode_dir,'w')
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
        assert(sent_num == len(content_list))
        assert(sent_num == len(pred_scores))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx][0])
            nbest = len(predict_results[idx])
            score_string = "# "
            for idz in range(nbest):
                score_string += format(pred_scores[idx][idz], '.4f')+" "
            fout.write(score_string.strip() + "\n")

            for idy in range(sent_length):
                try:  # Will fail with python3
                    label_string = content_list[idx][0][idy].encode('utf-8') + " "
                except:
                    label_string = content_list[idx][0][idy] + " "
                for idz in range(nbest):
                    label_string += predict_results[idx][idz][idy]+" "
                label_string = label_string.strip() + "\n"
                fout.write(label_string)
            fout.write('\n')
        fout.close()
        print("Predict %s %s-best result has been written into file. %s"%(name, nbest, self.decode_dir))


    def read_config(self,config_file):
        config = config_file_to_dict(config_file)
        ## read data:
        the_item = 'train_dir_1'
        if the_item in config:
            self.train_dir_1 = config[the_item]
        the_item = 'dev_dir_1'
        if the_item in config:
            self.dev_dir_1 = config[the_item]
        the_item = 'test_dir_1'
        if the_item in config:
            self.test_dir_1 = config[the_item]
        the_item = 'train_dir_2'
        if the_item in config:
            self.train_dir_2 = config[the_item]
        the_item = 'dev_dir_2'
        if the_item in config:
            self.dev_dir_2 = config[the_item]
        the_item = 'test_dir_2'
        if the_item in config:
            self.test_dir_2 = config[the_item]
        the_item = 'LM_dir_1'
        if the_item in config:
            self.LM_dir_1 = config[the_item]
        the_item = 'LM_dir_2'
        if the_item in config:
            self.LM_dir_2 = config[the_item]

        the_item = 'raw_dir_1'
        if the_item in config:
            self.raw_dir_1 = config[the_item]
        the_item = 'raw_dir_2'
        if the_item in config:
            self.raw_dir_2 = config[the_item]
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
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]
        the_item = 'task_emb_dir'
        if the_item in config:
            self.task_emb_dir = config[the_item]
        the_item = 'domain_emb_dir'
        if the_item in config:
            self.domain_emb_dir = config[the_item]
        the_item = 'task_emb_save_dir'
        if the_item in config:
            self.task_emb_save_dir = config[the_item]
        the_item = 'domain_emb_save_dir'
        if the_item in config:
            self.domain_emb_save_dir = config[the_item]


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
        the_item = 'norm_task_emb'
        if the_item in config:
            self.norm_task_emb = str2bool(config[the_item])
        the_item = 'norm_domain_emb'
        if the_item in config:
            self.norm_domain_emb = str2bool(config[the_item])


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
        the_item = 'sample_num'
        if the_item in config:
            self.LM_sample_num = int(config[the_item])

        ## read network:
        the_item = 'use_crf_sl'
        if the_item in config:
            self.use_crf_sl = str2bool(config[the_item])
        the_item = 'use_crf_lm'
        if the_item in config:
            self.use_crf_lm = str2bool(config[the_item])
        the_item = 'use_lm_sample'
        if the_item in config:
            self.LM_use_sample = str2bool(config[the_item])
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

        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item] ## feat_config is a dict






        ## read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        ## read Hyperparameters:
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

    def save_task_domain_embedding(self):
        if self.task_emb_dir is not None:
            with open('sample_data/task_emb.txt', 'w') as ft:
                for task, i in self.task_alphabet.iteritems():
                    ft.write(task)
                    ft.write(' ')
                    for t in self.pretrain_task_embedding[i]:
                        ft.write(str(round(t, 6)))
                        ft.write(' ')
                    ft.write('\n')
        if self.domain_emb_dir is not None:
            with open('sample_data/domain_emb.txt', 'w') as fd:
                for domain, i in self.domain_alphabet.iteritems():
                    fd.write(domain)
                    fd.write(' ')
                    for t in self.pretrain_domain_embedding[i]:
                        fd.write(str(round(t, 6)))
                        fd.write(' ')
                    fd.write('\n')





def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file,'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#',1)[0].split('=',1)
            item = pair[0]
            if item=="feature":
                if item not in config:
                    feat_dict = {}
                    config[item]= feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1,len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"]=conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"]=int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"]=str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated."%(pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False
