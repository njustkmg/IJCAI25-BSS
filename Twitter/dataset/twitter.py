import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.utils import pre_caption, clean_str
import os
import random
import torch
import numpy as np
import librosa
from torchvision import transforms
import glob, json, re, torch, en_vectors_web_lg, random

class dataTwitter(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words = 20):
        self.info = pd.read_csv(ann_file,sep='\t') # 读取了train/test.tsv中的内容
        self.transform = transform
        self.image_root = image_root # './data/Twitter15/twitter2015_images/'
        self.max_words = max_words

        # Loading all txt
        total_text_list = []
        ann_file1 = '/data/php_code/data_processing/Twitter15/annotations/train.tsv'
        ann_file2 = '/data/php_code/data_processing/Twitter15/annotations/dev.tsv'
        ann_file3 = '/data/php_code/data_processing/Twitter15/annotations/test.tsv'
        info1_ = pd.read_csv(ann_file1, sep='\t')
        info2_ = pd.read_csv(ann_file2, sep='\t')
        info3_ = pd.read_csv(ann_file3, sep='\t')
        info1 = info1_['String']
        info2 = info2_['String']
        info3 = info3_['String']
        for i in range(len(info1)):
            total_text_list.append(info1[i])
        for j in range(len(info2)):
            total_text_list.append(info2[j])
        for k in range(len(info3)):
            total_text_list.append(info3[k])

        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(total_text_list, use_glove=True)
        self.token_size = len(self.token_to_ix)

    def tokenize(self, stat_caps_list, use_glove=None):
        max_token = 0
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        for cap in stat_caps_list:
            words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()
            max_token = max(len(words), max_token)
            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token

    def proc_cap(self, cap, token_to_ix, max_token):
        cap_ix = np.zeros(max_token, np.int64)
        words = re.sub(r"([.,'!?\"()*#:;])", '', cap.lower()).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                cap_ix[ix] = token_to_ix[word]
            else:
                cap_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return cap_ix

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        index = int(index)  # 确保索引是整数
        if index >= len(self.info) or index < 0:
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.info)}")

        label = self.info["Label"][index]
        label = int(label)
        # bert
        text1 = self.info['String'][index]
        text2 = clean_str(text1) # 文本清洗：去除一些符号
        text = pre_caption(text2, self.max_words) # # 文本清洗：大写变小写

        # resnet
        ID = self.info["ImageID"][index]
        imagePath = self.image_root + self.info["ImageID"][index]
        image = Image.open(imagePath).convert('RGB')
        image = self.transform(image)

        return image, text, label

