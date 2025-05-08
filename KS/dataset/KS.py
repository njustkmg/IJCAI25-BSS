import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torch
import numpy as np
import librosa
from torchvision import transforms


class VADataset(Dataset):
    '''
       初始化
       model = 'train
       config中参数 train_file 训练集csv文件; test_file 测试集csv文件; data_path数据集根目录
       返回
       spectrogram shape 257 X 1004
       images shape CTHW : 3 X 3 X 256 X 256
       label
    '''

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.use_pre_frame = 3

        '''
        train部分已修改
        '''
        train_video_data, train_audio_data, train_label, train_class = [], [], [], []
        test_video_data, test_audio_data, test_label, test_class = [], [], [], []
        weight_video_data, weight_audio_data, weight_label, weight_class = [], [], [], []
        root = args.data_root

        train_file = os.path.join(root, 'annotations', 'train.csv')
        data = pd.read_csv(train_file)
        # data = data.head(100)
        self.labels = data['label']
        self.files = data['youtube_id']
        for i, item in enumerate(self.files):
            video_dir = os.path.join(root, 'train_img', 'Image-01-FPS', item)
            audio_dir = os.path.join(root, 'train_wav', item + '.wav')
            if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                train_video_data.append(video_dir)
                train_audio_data.append(audio_dir)
                if self.labels[i] not in train_class:
                    train_class.append(self.labels[i])
                train_label.append(self.labels[i])
        '''
        test部分已修改
        '''
        test_file = os.path.join(root, 'annotations', 'test.csv')
        data = pd.read_csv(test_file)
        self.labels = data['label']
        self.files = data['youtube_id']
        for i, item in enumerate(self.files):
            video_dir = os.path.join(root, 'test_img', 'Image-01-FPS', item)
            audio_dir = os.path.join(root, 'test_wav', item + '.wav')
            if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                test_video_data.append(video_dir)
                test_audio_data.append(audio_dir)
                if self.labels[i] not in test_class:
                    test_class.append(self.labels[i])
                test_label.append(self.labels[i])
        # assert len(train_class) == len(test_class)
        '''
        weight 部分已修改
        '''
        weight_file = os.path.join(root, 'annotations', 'weight.csv')
        data = pd.read_csv(weight_file)
        self.labels = data['label']
        self.files = data['youtube_id']
        root = args.data_root
        for i, item in enumerate(self.files):
            video_dir = os.path.join(root, 'train_img', 'Image-01-FPS', item)
            audio_dir = os.path.join(root, 'train_wav', item + '.wav')
            if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                weight_video_data.append(video_dir)
                weight_audio_data.append(audio_dir)
                if self.labels[i] not in weight_class:
                    weight_class.append(self.labels[i])
                weight_label.append(self.labels[i])

        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))

        if mode == 'train':
            self.video = train_video_data
            self.audio = train_audio_data
            self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
        if mode == 'weight':
            self.video = weight_video_data
            self.audio = weight_audio_data
            self.label = [class_dict[weight_label[idx]] for idx in range(len(weight_label))]
        if mode == 'test':
            self.video = test_video_data
            self.audio = test_audio_data
            self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=35400, mono=True)
        if len(sample) == 0:
            sample = np.array([0])
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)
            # sample = np.tile(sample, 10)
        start_point = 0
        new_sample = sample[start_point:start_point + rate * 10]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.
        spectrogram = librosa.stft(new_sample, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        # print(np.shape(spectrogram))
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.video[idx])
        if self.mode == 'train':
            select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
            select_index.sort()
        else:
            select_index = [idx for idx in range(0, len(image_samples), len(image_samples) // self.use_pre_frame)]
            # select_index = np.random.choice(len(image_samples), size=self.use_pre_frame, replace=False)
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))

        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.video[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = images.permute((1, 0, 2, 3))
        # label
        # one_hot = np.eye(31)
        # one_hot_label = one_hot[self.label[idx]]
        # label = torch.FloatTensor(one_hot_label)

        label = self.label[idx]
        return spectrogram, images, label