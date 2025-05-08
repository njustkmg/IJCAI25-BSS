from models.Model3D import InceptionI3d
from models.Resnet import resnet18

import torch.nn as nn
import torch


class RGBEncoder(nn.Module):
    def __init__(self, config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out  # BxNx2048


class OFEncoder(nn.Module):
    def __init__(self, config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/flow_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.ofmodel = model

    def forward(self, x):
        out = self.ofmodel(x)
        return out  # BxNx2048


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        # download the checkpoint from https://github.com/piergiaj/pytorch-i3d/tree/master/models
        # https://github.com/piergiaj/pytorch-i3d/tree/master
        pretrained_dict = torch.load('/data/hlf/imbalance/unimodal/checkpoint/rgb_imagenet.pt')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.depthmodel = model

    def forward(self, x):
        out = self.depthmodel(x)
        return out  # BxNx2048


class TriNet(nn.Module):
    def __init__(self, args, mask_model=1, act_fun=nn.GELU()):
        super(TriNet, self).__init__()
        self.rgb_encoder = RGBEncoder(args)
        self.of_encoder = OFEncoder(args)
        self.depth_encoder = DepthEncoder(args)
        self.cls_m1 = nn.Linear(1024, 25)
        self.cls_m2 = nn.Linear(1024, 25)
        self.cls_m3 = nn.Linear(1024, 25)
        self.cls_mm = nn.Linear(1024*3, 25)

    def forward(self, rgb, of, depth):
        # 获取音频和视频的特征
        # rgb:(2,3,64,224,224)
        # of:(2,2,80,224,224)
        # depth:(2,1,64,224,224)
        rgb_feature = self.rgb_encoder(rgb) # (bs,1024)
        of_feature = self.of_encoder(of) # (bs, 1024)
        depth_feature = self.depth_encoder(depth) # (bs,1024)
        out_m1 = self.cls_m1(rgb_feature)
        out_m2 = self.cls_m2(of_feature)
        out_m3 = self.cls_m3(depth_feature)
        out_mm = self.cls_mm(torch.cat((rgb_feature, of_feature, depth_feature), dim=1))


        return out_m1, out_m2, out_m3, out_mm