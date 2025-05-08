import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet50
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, TwoLayerCrossAttention, DeepFusion, AdvancedAdapterModule
import numpy as np

class AVNet(nn.Module):
    def __init__(self, args):
        super(AVNet, self).__init__()

        n_classes = 31
        self.linear_a = nn.Linear(512, n_classes)
        self.linear_v = nn.Linear(512, n_classes)
        self.linear_last = nn.Linear(1024, n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')


    def forward(self, audio, visual):

        # (64,1,257,1003) (64,3,3,354,354)
        a = self.audio_net(audio) # (64,512,9,32)
        a = F.adaptive_avg_pool2d(a, 1) # (64,512,1,1)
        a = torch.flatten(a, 1) # (64,512)

        v = self.visual_net(visual) # (192,512,12,12)
        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4) # (64,512,3,12,12)
        v = F.adaptive_avg_pool3d(v, 1)  # (64,512,1,1,1)
        v = torch.flatten(v, 1)  # (64,512)

        out1 = self.linear_a(a)
        out2 = self.linear_v(v)
        out = self.linear_last(torch.cat((a, v), dim=1))

        # return out, a, v
        return out, out1, out2, a, v
