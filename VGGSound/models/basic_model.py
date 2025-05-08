import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet50

class AVNet(nn.Module):
    def __init__(self, args):
        super(AVNet, self).__init__()

        n_classes = 310
        self.linear_a = nn.Linear(512, n_classes)
        self.linear_v = nn.Linear(512, n_classes)
        self.linear_last = nn.Linear(1024, n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')


    def forward(self, audio, visual):

        # (64,1,257,502) (64,3,1,384,384)
        a = self.audio_net(audio) # (64,512,9,16)
        a = F.adaptive_avg_pool2d(a, 1) # (64,512,1,1)
        a = torch.flatten(a, 1) # (64,512)

        v = self.visual_net(visual) # (64,512,12,12)
        v = F.adaptive_avg_pool2d(v, 1) # (64,512,1,1)
        v = torch.flatten(v, 1) # (64,512)

        out_m1 = self.linear_a(a) # 309
        out_m2 = self.linear_v(v) # 309
        out_mm = self.linear_last(torch.cat((a, v), dim=1)) # 309

        return out_mm, out_m1, out_m2







