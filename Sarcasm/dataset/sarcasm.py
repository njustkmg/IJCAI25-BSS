from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torch
import numpy as np
import librosa
from torchvision import transforms
import glob, json, re, torch, en_vectors_web_lg, random
from dataset.randaugment import RandomAugment

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
class dataSarcasm(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.cls_num = 2
        self.labels = []
        for i in self.ann:
            self.labels.append(int(i['label']))

    def __len__(self):
        return len(self.ann)

    def get_num_classes(self):
        return self.cls_num

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['text'], self.max_words)
        target = int(ann['label'])
        return image, caption, target

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])