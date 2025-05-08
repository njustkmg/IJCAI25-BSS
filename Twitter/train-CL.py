import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pdb
from dataset.twitter import dataTwitter
from models.basic_model import ITNet
from utils.utils import setup_seed, weight_init
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from torchvision import transforms
from transformers import BertModel, BertTokenizer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', default=1, type=int)

    parser.add_argument('--audio_path', default='/data/wfq/paper/dataset/CREMA/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/data/wfq/paper/dataset/CREMA/', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=113, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='4', type=str, help='GPU ids')

    return parser.parse_args()

def train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train):

    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    print('Start Training')
    pbar = tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.train()

    tokenizer = BertTokenizer.from_pretrained('/data/hlf/imbalance/unimodal/bert-base-uncased')
    for step, (image, text, label) in enumerate(train_dataloader):
        image = image.to(device)
        text = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out_mm, out_m1, out_m2 = net(image.unsqueeze(2).float(), text)
        loss1 = criterion(out_mm, label)
        loss2 = criterion(out_m1, label)
        loss3 = criterion(out_m2, label)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        _total_loss += loss.item()

        pbar.set_postfix(**{'train_loss': _total_loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)

    pbar.close()
    scheduler.step() # 学习率更新

    return _total_loss / epoch_step_train

def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    _loss = 0

    # 用于收集所有样本的预测结果和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', mininterval=0.3, ncols=120)
    net.eval()

    tokenizer = BertTokenizer.from_pretrained('/data/hlf/imbalance/unimodal/bert-base-uncased')
    for step, (image, text, label) in enumerate(test_dataloader):
        with torch.no_grad():
            image = image.to(device)
            text = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
            label = label.to(device)

            out_mm, out_m1, out_m2 = net(image.unsqueeze(2).float(), text)

            loss_mm = criterion(out_mm, label)
            loss_m1 = criterion(out_m1, label)
            loss_m2 = criterion(out_m2, label)
            loss = loss_mm + loss_m1 + loss_m2
            _loss += loss.item()

            preds = torch.max(out_mm+out_m1+out_m2, dim=1)[1]  # 使用out直接获得预测类别
            all_preds.extend(preds.tolist())
            all_labels.extend(label.tolist())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 注意转换为NumPy数组用于性能计算
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average='macro')

    return _loss / epoch_step_test, accuracy, f1_macro

def normalize_min_max(values):
    """最小-最大归一化，将输入列表归一化到 [0, 1]"""
    min_val = np.min(values)
    max_val = np.max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

def model_evaluate(model, train_dataset, device):
    similarities = []
    losses = []
    tokenizer = BertTokenizer.from_pretrained('/data/hlf/imbalance/unimodal/bert-base-uncased')
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不进行 reduction，保留每个样本的 loss
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)

    model.train()  # 设置模型为训练模式，以便计算梯度

    for image, text, label in dataloader:
        image = image.to(device)
        text = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
        label = label.to(device)

        model.zero_grad()  # 清除前一个批次的梯度
        out_mm, out_m1, out_m2 = model(image.unsqueeze(2).float(), text)  # 正向传播

        # 计算单模态分类头的余弦相似度
        out_m1_ = F.softmax(out_m1, dim=1)
        out_m2_ = F.softmax(out_m2, dim=1)
        cosine_sim = F.cosine_similarity(out_m1_, out_m2_, dim=1)  # 计算每个样本的相似度 (shape: [batch_size])

        # 使用 detach() 分离计算图并将 Tensor 转为 NumPy 数组
        similarities.extend(cosine_sim.detach().cpu().numpy())  # 分离并保存每个样本的余弦相似度

        # 计算每个样本的损失 (批处理情况下)
        loss_m1 = criterion(out_m1, label)  # (shape: [batch_size])
        loss_m2 = criterion(out_m2, label)  # (shape: [batch_size])
        loss_mm = criterion(out_mm, label)  # (shape: [batch_size])
        total_loss = loss_m1 + loss_m2 + loss_mm  # 对应每个样本的总损失 (shape: [batch_size])
        losses.extend(total_loss.detach().cpu().numpy())  # 分离计算图并保存每个样本的损失

    # 归一化 similarities 和 losses 到 [0, 1] 范围
    similarities_norm = normalize_min_max(similarities)
    losses_norm = normalize_min_max(losses)

    # 结合归一化后的余弦相似度和损失来确定样本难度
    combined_scores = [(1 - sim) + loss for sim, loss in zip(similarities_norm, losses_norm)]  # 相似度低 + 损失大 = 更难

    # 根据综合得分排序，从易到难
    sorted_indices = sorted(range(len(combined_scores)), key=lambda k: combined_scores[k], reverse=False) # False是CL

    return sorted_indices


if __name__ == '__main__':
    # ---------------------参数----------------------
    args = get_arguments()
    print(args)

    # ---------------------设备----------------------
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print('GPU设备数量为:', torch.cuda.device_count())
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    # ---------------------模型----------------------
    net = ITNet(args)
    net.to(device) # 将模型在指定的device上进行初始化，这里是3号GPU，索引为0号
    net = torch.nn.DataParallel(net, device_ids=gpu_ids) # 对模型进行封装，分发到多个GPU上运行
    net.cuda()

    # ---------------------数据----------------------
    train_csv = '/data/php_code/data_processing/Twitter15/annotations/train.tsv'
    val_csv = '/data/php_code/data_processing/Twitter15/annotations/test.tsv'
    imageroot = '/data/php_code/data_processing/Twitter15/twitter2015_images/'
    trans = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])

    train_dataset = dataTwitter(train_csv, trans, imageroot)
    test_dataset = dataTwitter(val_csv, trans, imageroot)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32, pin_memory=True)
    print('训练数据量: ', len(train_dataset))
    print('测试数据量: ', len(test_dataset))
    epoch_step_train = len(train_dataset) // train_dataloader.batch_size
    epoch_step_test = math.ceil(len(test_dataset) / test_dataloader.batch_size)  # 因为验证集没有drop_last，所以多一个step，向上取整

    # --------------------优化器---------------------
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    # ------------------训练and验证-------------------
    if True:
        best_acc = 0.0
        best_acc = 0
        lambda_0 = 0.1
        T_grow = 40

        results_file = './results/CL-PreSim+Loss.txt'
        if os.path.exists(results_file):
            os.remove(results_file)

        for epoch in range(args.epochs):
            if epoch in [0]:
                print(f'第{epoch + 1}个epoch开始多模态评估！')
                sorted_indices = model_evaluate(net, train_dataset, device)
                print(f'第{epoch + 1}个epoch结束多模态评估！')

            lambda_t = min(1, math.sqrt((1 - lambda_0 ** 2) / T_grow * (epoch + 1) + lambda_0 ** 2))
            current_num_samples = int(len(sorted_indices) * lambda_t)
            subset_indices = sorted_indices[:current_num_samples]
            subset = torch.utils.data.Subset(train_dataset, subset_indices)
            current_dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

            mean_loss_train = train(args, epoch, net, device, current_dataloader, optimizer, scheduler, epoch_step_train)
            mean_loss_test, test_acc, test_f1 = test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test)

            print('********************************************************************')
            print('Epoch:' + str(epoch + 1) + '/' + str(args.epochs))
            print('Now train_loss: %.4f || Now test_loss: %.4f' % (mean_loss_train, mean_loss_test))
            print('Now test_acc: %.4f || Now test_f1: %.4f' % (test_acc, test_f1))

            with open(results_file, 'a') as f:
                f.write(f'{epoch + 1} {test_acc:.4f} {test_f1:.4f}\n')

            if test_acc > best_acc:
                best_acc = float(test_acc)
                best_acc_epoch = epoch + 1

            print('Best Accuracy: %.4f, Best Epoch: %d' % (best_acc, best_acc_epoch))
            print('********************************************************************')





