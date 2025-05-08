import torch
import torch.nn as nn
import numpy as np
import random

def setup_seed1(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

import re
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import csv
import math
from functools import partial

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


def pad_or_trim(tensor, max_length):
    if tensor.size(0) > max_length:
        return tensor[:max_length]
    elif tensor.size(0) < max_length:
        padding_size = max_length - tensor.size(0)
        return torch.cat([tensor, torch.zeros(padding_size, dtype=tensor.dtype)], dim=0)
    else:
        return tensor


def cla_alpha1(alpha1, epoch, max_epochs):
    sigmoid_decay = lambda x: 1 / (1 + np.exp(-x))

    normalized_epoch = epoch / max_epochs

    updated_alpha1 = alpha1 * sigmoid_decay(10 * (normalized_epoch - 0.5))

    return updated_alpha1


def get_neg_sample(texts, neg_index):
    texts["input_ids"] = texts["input_ids"][neg_index]
    texts["token_type_ids"] = texts["token_type_ids"][neg_index]
    texts["attention_mask"] = texts["attention_mask"][neg_index]
    return texts


def calculate_matrix(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    f1 = f1_score(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return f1, acc


def clean_str(text):
    # 使用正则表达式匹配英文单词
    english_words = re.findall(r'\b[A-Za-z]+\b', text)
    # 将匹配到的英文单词连接成字符串
    result_text = ' '.join(english_words)
    return result_text


def add_data_to_tsv(alpha_i2t, alpha_t2i):
    try:
        with open("./logs/loss_alpha.tsv", mode="r") as file:
            reader = csv.reader(file, delimiter='\t')
            header = next(reader)
    except FileNotFoundError:
        header = ["alpha_i2t", "alpha_t2i"]
    with open("./logs/loss_alpha.tsv", mode="a", newline="") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow([alpha_i2t, alpha_t2i])

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
