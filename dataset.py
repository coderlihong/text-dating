import random
from collections import Counter

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
import torch
import pandas as pd
import xml.etree.ElementTree as ET
import os
from sklearn.model_selection import KFold


def get_weight():
    weight_list = []
    for time in ['西汉', '东汉', '南朝宋', '西晋', '唐', '南朝梁', '北朝齐', '后晋', '宋', '元', '明',
                 '清']:
        tmp_weight = torch.load(f'./single_train_weight/mean/{time}.pth',
                                map_location='cpu')
        tmp_weight.require_grad = False
        weight_list.append(tmp_weight)
    weight_list = torch.stack(weight_list, dim=0)

    return weight_list


def getdata(mode, max_sentence_len, max_tokens_len, path):
    corpus = []
    labels = []
    label2idx = {}

    df = pd.read_excel(path + mode + '.xlsx')

    labeldf = list(df['time'])

    file_list = os.listdir(
        path + mode + '/')


    file_list = sorted(file_list)
    total_len = []
    for file_name in file_list:

        tree = ET.parse(
            path + mode + "/" + file_name)

        file_num = file_name.split('.')[0]

        root = tree.getroot()
        sentence = root.findall('sentence')
        outdata = []
        tmp_len = 0
        for i in sentence:
            tokens = list(i.text)
            if len(tokens) <= max_tokens_len:
                outdata.append(tokens)
                tmp_len += len(tokens)
            if len(outdata) > max_sentence_len - 1:
                corpus.append(outdata)
                total_len.append(tmp_len)
                tmp_len = 0
                labels.append(labeldf[int(file_num) - 1])
                if labeldf[int(file_num) - 1] not in label2idx:
                    label2idx[labeldf[int(file_num) - 1]] = len(label2idx)
                outdata = []
        # corpus.append(outdata)
        # labels.append(labeldf[int(file_num) - 1])

    mean_len = np.array(total_len).mean()
    max_len = np.max(total_len)
    print(f'mean_len:{mean_len}')
    print(f'max_len:{max_len}')
    return corpus, labels, label2idx


class Mydataset(Dataset):
    def __init__(self, corpus, labels, tokenizer, vec_list, label2idx, max_sentence_len, max_tokens_len, mode):
        self.corpus = corpus
        self.labels = labels
        self.tokenizer = tokenizer
        self.vec_list = {

        }
        for i, j in zip(label2idx, vec_list):
            self.vec_list[i] = j

        self.label2idx = label2idx
        self.max_length_sentences = max_sentence_len
        self.max_length_word = max_tokens_len
        self.special_tokens_mask = None
        self.mlm_probability = 0.15
        self.prob_replace_mask = 0.8
        self.prob_replace_rand = 0.1
        self.prob_keep_ori = 0.1
        self.mode = mode

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):

        corpus = self.corpus[idx]
        labels = self.labels[idx]
        vector = self.vec_list[labels]
        all_sentence = ''
        for i in corpus:
            all_sentence += ''.join(i)
        document_encode = self.tokenizer.encode_plus(all_sentence, max_length=512, padding='max_length',
                                                     add_special_tokens=True,
                                                     truncation=True)

        sentence_attention = [0] * len(document_encode) + [-torch.inf] * (
                self.max_length_sentences - len(document_encode))

        target = [vector[word].tolist() for word in document_encode]

        target = torch.tensor(target)
        document_encode = torch.tensor(document_encode)

        sentence_attention = torch.tensor(sentence_attention, dtype=torch.float)
        label = torch.tensor(self.label2idx[labels])
        features = document_encode.clone().view(-1)
        if self.mode == 'train':
            document_encode, mlm_lables = self.mask_tokens(features)
            document_encode = document_encode.view(self.max_length_sentences, self.max_length_word)

            return document_encode, sentence_attention, target, label, mlm_lables
        else:
            return document_encode, sentence_attention, target, label

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if self.special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        else:
            special_tokens_mask = self.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.prob_replace_rand / (1 - self.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class MyMLMdataset(Dataset):
    def __init__(self, corpus, labels, tokenizer, vec_list, label2idx, max_sentence_len, max_tokens_len, mode, word):
        self.corpus = corpus
        self.labels = labels
        self.tokenizer = tokenizer
        self.vec_list = {

        }
        for i, j in zip(label2idx, vec_list):
            self.vec_list[i] = j

        self.label2idx = label2idx
        self.max_length_sentences = max_sentence_len
        self.max_length_word = max_tokens_len
        self.special_tokens_mask = None
        self.mlm_probability = 0.15
        self.prob_replace_mask = 0.8
        self.prob_replace_rand = 0.1
        self.prob_keep_ori = 0.1
        self.mode = mode
        self.word = word

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):

        corpus = self.corpus[idx]
        labels = self.labels[idx]
        vector = self.vec_list[labels]

        document_encode = [
            self.tokenizer.encode(''.join(sentences).replace(self.word, '[MASK]'), max_length=32, padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True) for sentences in corpus]

        sentence_attention = [0] * len(document_encode) + [-torch.inf] * (
                self.max_length_sentences - len(document_encode))

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[101, 102] + [0 for _ in range(self.max_length_word - 2)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        target = [[vector[word].tolist() for word in sentence] for sentence in document_encode]

        target = torch.tensor(target)
        document_encode = torch.tensor(document_encode)

        sentence_attention = torch.tensor(sentence_attention, dtype=torch.float)
        label = torch.tensor(self.label2idx[labels])
        features = document_encode.clone().view(-1)
        if self.mode == 'train':
            document_encode, mlm_lables = self.mask_tokens(features)
            document_encode = document_encode.view(self.max_length_sentences, self.max_length_word)

            return document_encode, sentence_attention, target, label, mlm_lables
        else:
            return document_encode, sentence_attention, target, label

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if self.special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        else:
            special_tokens_mask = self.special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        current_prob = self.prob_replace_rand / (1 - self.prob_replace_mask)
        indices_random = torch.bernoulli(
            torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def load_data(config, tokenzier, k = None):
    label2idx = {}
    weight = get_weight()

    if k !=None:
        train_idx = np.load('./k_idx/' + str(k) + 'train.npy')
        valid_idx = np.load('./k_idx/' + str(k) + 'valid.npy')
        train_corpus, train_labels, train_label2idx = getdata('train', config.max_sentence_len, config.max_tokens_len,
                                                              config.base_data_path)
        label2idx = train_label2idx



        valid_corpus, valid_labels, _ = getdata('val', config.max_sentence_len, config.max_tokens_len,
                                                config.base_data_path)
        corpus, labels = train_corpus + valid_corpus, train_labels + valid_labels
        train_corpus = [corpus[idx] for idx in train_idx]
        train_labels = [labels[idx] for idx in train_idx]

        valid_corpus = [corpus[idx] for idx in valid_idx]
        valid_labels = [labels[idx] for idx in valid_idx]
    else:
        train_corpus, train_labels, train_label2idx = getdata('train', config.max_sentence_len, config.max_tokens_len,
                                                              config.base_data_path)
        label2idx = train_label2idx

        valid_corpus, valid_labels, _ = getdata('val', config.max_sentence_len, config.max_tokens_len,
                                                config.base_data_path)
    train_set = Mydataset(train_corpus, train_labels, tokenzier, weight, label2idx, config.max_sentence_len,
                          config.max_tokens_len,
                          'train')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, pin_memory=True, num_workers=0, shuffle=True)
    val_set = Mydataset(valid_corpus, valid_labels, tokenzier, weight, label2idx, config.max_sentence_len, config.max_tokens_len,
                        'valid')
    val_loader = DataLoader(val_set, batch_size=config.batch_size, pin_memory=True, num_workers=0, shuffle=False)

    corpus, labels, _ = getdata('test', config.max_sentence_len, config.max_tokens_len, config.base_data_path)
    test_set = Mydataset(corpus, labels, tokenzier, weight, label2idx, config.max_sentence_len, config.max_tokens_len,
                         'test')
    test_loader = DataLoader(test_set, batch_size=config.batch_size, pin_memory=True, num_workers=0, shuffle=False)
    corpus = [k for i in corpus for j in i for k in j]
    word_freq = Counter(corpus)

    return train_loader, val_loader, test_loader, weight, label2idx, word_freq


def load_word_data(config, tokenizer, word, label2idx):
    if label2idx == None:
        corpus, labels, train_label2idx = getdata('train', config.max_sentence_len, config.max_tokens_len,
                                                  config.base_data_path)
        label2idx = train_label2idx
    corpus, labels, _ = getdata('test', config.max_sentence_len, config.max_tokens_len, config.base_data_path)
    weight = get_weight()

    new_corpus = []
    new_labels = []
    for i, j in zip(corpus, labels):
        for k in i:
            if word in k:
                new_corpus.append(k)
                new_labels.append(j)
                break

    test_set = MyMLMdataset(new_corpus, new_labels, tokenizer, weight, label2idx, config.max_sentence_len,
                            config.max_tokens_len, 'test', word)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, pin_memory=True, num_workers=0, shuffle=False)
    return test_loader, label2idx


def getCorpusWithWord(word, mode, max_sentence_len, max_tokens_len, path):
    corpus = []
    labels = []
    label2idx = {}
    df = pd.read_excel(path + mode + '.xlsx')

    labeldf = list(df['time'])

    file_list = os.listdir(
        path + mode + '/')


    file_list = sorted(file_list)
    total_len = []
    for file_name in file_list:

        tree = ET.parse(
            path + mode + "/" + file_name)


        file_num = file_name.split('.')[0]

        root = tree.getroot()
        sentence = root.findall('sentence')
        outdata = []

        for i in sentence:
            tokens = list(i.text)
            if len(tokens) <= max_tokens_len:
                outdata.append(tokens)

            if len(outdata) > max_sentence_len - 1:
                corpus.append(outdata)

                labels.append(labeldf[int(file_num) - 1])
                if labeldf[int(file_num) - 1] not in label2idx:
                    label2idx[labeldf[int(file_num) - 1]] = len(label2idx)
                outdata = []
        # corpus.append(outdata)
        # labels.append(labeldf[int(file_num) - 1])
        new_corpus = []
        new_labels = []
        for c, label in zip(corpus, labels):
            tmp = '<sep>'.join([''.join(sentence) for sentence in c])
            if word in tmp:
                new_corpus.append(tmp)
                new_labels.append(label)
        fout = open('test_corpus.txt', mode='w', encoding='utf8')
        for i, j in zip(new_corpus, new_labels):
            fout.write(i + '\t' + j + '\n')


if __name__ == '__main__':
    # base_data_path = '/home/wh161054332/project/documentcls/corpus_data/'
    class Config():
        def __init__(self):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained('D:/code/pretrain_models/bert-base-chinese')
            #
            # self.tokenizer = AutoTokenizer.from_pretrained('/home/wh161054332/pretrain/bert-ancient-chinese')
            self.embedding_size = 768
            self.lr = 1e-5
            self.class_num = 12
            self.max_sentence_len = 30
            self.max_tokens_len = 32
            self.batch_size = 32
            self.epochs = 20
            # self.base_data_path = 'D:/code/pythoncode/ancient_chinese/Temporal/Hierarchical-attention-networks-pytorch-master/data/'
            self.base_data_path = 'D:/code/pycode/documentcls/corpus_data/'
            self.gru_size = 100
            self.layers = 3
            self.pretrain_path = 'D:/code/Pretrain_models/bert-base-chinese'
            self.filter_sizes = (2, 3, 4, 5)  # 卷积核尺寸
            self.num_filters = 256  # 卷积核数量(channels数)
            self.dropout = 0.1


    config = Config()
    corpus, train_labels, label2idx = getdata('train', 30, 32, config.base_data_path)
    train_corpus = corpus
    valid_corpus, valid_labels, _ = getdata('val', 30, 32, config.base_data_path)
    corpus, labels = corpus + valid_corpus, train_labels + valid_labels
    test_corpus, test_labels, _ = getdata('test', 30, 32, config.base_data_path)
    print(len(train_corpus))
    print(len(valid_corpus))
    print(len(test_corpus))
    shunxu = ['西汉', '东汉', '西晋', '南朝宋', '南朝梁', '北朝齐','唐', '后晋', '宋', '元', '明', '清']
    trainmycount = {item:0 for item in shunxu}
    for i,j in zip(train_corpus,train_labels):
        trainmycount[j]+=1
    print(trainmycount)
    validmycount = {item:0 for item in shunxu}
    for i,j in zip(valid_corpus,valid_labels):
        validmycount[j]+=1
    print(validmycount)
    testmycount = {item:0 for item in shunxu}
    for i,j in zip(test_corpus,test_labels):
        testmycount[j]+=1
    print(testmycount)
    # 将数据集分成5个折叠，进行交叉验证
    kfold = KFold(n_splits=5, shuffle=True)
    total_len = [i for i in range(0, len(train_corpus))]
    raw_idx = [i for i in range(30673, len(corpus))]
    print(raw_idx)

    # 对于每个fold，获取训练集和测试集
    start_k = 1
    for train_idx, test_idx in kfold.split(total_len):
        trainidx = np.append(train_idx,raw_idx)
        add_list = random.sample(test_idx.tolist(),2406)
        trainidx = np.append(trainidx,add_list)
        test_idx = [item for item in test_idx if item not in add_list]
        print(len(trainidx))
        print(len(test_idx))
        np.save('./k_idx/' + str(start_k) + 'train.npy', np.array(trainidx))
        np.save('./k_idx/' + str(start_k) + 'valid.npy', np.array(test_idx))

        tmp = np.load('./k_idx/' + str(start_k) + 'train.npy')
        print(tmp)
        start_k += 1

