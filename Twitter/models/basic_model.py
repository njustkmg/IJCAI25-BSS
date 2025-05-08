import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18, resnet50, TextEncoder


class ITNet(nn.Module):
    def __init__(self, args):
        super(ITNet, self).__init__()

        self.linear_txt = nn.Linear(2048, 3)
        self.linear_img = nn.Linear(2048, 3)
        self.visual_net = resnet50(modality='visual')
        checkpoint = torch.load('/data/hlf/imbalance/unimodal/checkpoint/resnet50-0676ba61.pth')
        print(self.visual_net.load_state_dict(checkpoint, strict=False))
        self.text_net = TextEncoder("/data/hlf/imbalance/unimodal/bert-base-uncased")
        self.linear_last = nn.Linear(4096, 3)
        # self.linear_last = nn.Linear(2816, 3)
        self.fc = nn.Linear(768, 2048)

    def forward(self, image, text):

        # (64,3,1,384,384) BatchEncoding:3
        img = self.visual_net(image) # (64,2048,12,12)
        img = F.adaptive_avg_pool2d(img, 1)  # (64,2048,1,1)
        img = torch.flatten(img, 1)  # (64,2048)

        txt = self.text_net(text)  # (64,768)
        txt = self.fc(txt)

        out_m1 = self.linear_img(img) # (64,3)
        out_m2 = self.linear_txt(txt) # (64,3)
        out_mm = self.linear_last(torch.cat((img, txt), dim=1)) # (64,3)

        return out_mm, out_m1, out_m2





