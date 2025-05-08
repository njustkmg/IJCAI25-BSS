import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertTokenizer
import os, math

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'



class ITNet(nn.Module):
    def __init__(self, args):
        super(ITNet, self).__init__()

        self.visual_net = torchvision.models.resnet50()
        checkpoint = torch.load('/data/hlf/imbalance/unimodal/checkpoint/resnet50-0676ba61.pth')
        self.visual_net.load_state_dict(checkpoint)
        self.text_net = BertModel.from_pretrained('/data/hlf/imbalance/unimodal/bert-base-uncased',
                                                      add_pooling_layer=False)

        self.fix = nn.Linear(1000, 768)
        self.cls_shared = nn.Linear(768, 2)
        self.cls_mm = nn.Linear(768*2, 2)


    def forward(self, image, text):
        img = self.visual_net(image)  # (32,1000)
        img = self.fix(img) # (32,768)
        txt = self.text_net(text.input_ids,
                                             attention_mask=text.attention_mask,
                                             return_dict=True
                                             ).last_hidden_state[:,0,:] # (32,768)


        out_mm = self.cls_mm(torch.cat((img, txt), dim=1))
        out_m1 = self.cls_shared(img)
        out_m2 = self.cls_shared(txt)

        return out_mm, out_m1, out_m2






