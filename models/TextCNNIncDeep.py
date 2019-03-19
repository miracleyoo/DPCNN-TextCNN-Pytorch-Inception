# coding: utf-8
# Author: Miracle Yoo

from .BasicModule import BasicModule
import torch
import torch.nn as nn
from copy import deepcopy
torch.manual_seed(1)


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class TextCNNIncDeep(BasicModule):
    def __init__(self, opt):
        """
        Warning! This module is NOT a pure DPCNN! It is a conbination of TextCNN and DPCNN,
        which means it will use TextCNN as its head feature extraction part, and use DPCNN
        to dig its high-level features accordingly.  
        """
        super(TextCNNIncDeep, self).__init__()
        self.model_name = "TextCNNIncDeep"
        self.opt = opt

        if opt.USE_CHAR:
            # use char instead of word
            opt.VOCAB_SIZE = opt.CHAR_SIZE

        self.encoder = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)

        question_convs1 = [nn.Sequential(
                nn.Conv1d(in_channels=opt.EMBEDDING_DIM,
                          out_channels=opt.TITLE_DIM,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(opt.TITLE_DIM),
                nn.ReLU(inplace=True),

                nn.MaxPool1d(kernel_size=(opt.SENT_LEN - kernel_size + 1))
            )for kernel_size in opt.SIN_KER_SIZE]

        question_convs2 = [nn.Sequential(
                nn.Conv1d(in_channels=opt.EMBEDDING_DIM,
                          out_channels=opt.TITLE_DIM,
                          kernel_size=kernel_size[0]),
                nn.BatchNorm1d(opt.TITLE_DIM),
                nn.ReLU(inplace=True),

                nn.Conv1d(in_channels=opt.TITLE_DIM,
                          out_channels=opt.TITLE_DIM,
                          kernel_size=kernel_size[1]),
                nn.BatchNorm1d(opt.TITLE_DIM),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=(opt.SENT_LEN - kernel_size[0] - kernel_size[1] + 2))
            )for kernel_size in opt.DOU_KER_SIZE]

        question_convs = question_convs1
        question_convs.extend(question_convs2)

        self.num_seq = len(opt.DOU_KER_SIZE) + len(opt.SIN_KER_SIZE)
        self.change_dim_conv  = nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)

        self.question_convs = nn.ModuleList(question_convs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=opt.TITLE_DIM),
            nn.ReLU(),
            nn.Conv1d(opt.TITLE_DIM, opt.NUM_ID_FEATURE_MAP,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP,
                      kernel_size=3, padding=1)
        )
        resnet_block_list = []
        while (self.num_seq > 2):
            resnet_block_list.append(ResnetBlock(opt.NUM_ID_FEATURE_MAP))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(opt.NUM_ID_FEATURE_MAP*self.num_seq, opt.NUM_CLASSES),
            nn.BatchNorm1d(opt.NUM_CLASSES),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES)
        )

    def forward(self, question):
        question = self.encoder(question)
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        x = [question_conv(question.permute(0, 2, 1))
             for question_conv in self.question_convs]
        x = torch.cat(x, dim=2)
        xp = x
        xp = self.change_dim_conv(xp)
        x = self.conv(x)
        x = x+xp
        x = self.resnet_layer(x)
        x = x.view(self.opt.BATCH_SIZE, -1)
        x = self.fc(x)
        return x
