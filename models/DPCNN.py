# coding: utf-8
# Author: Miracle Yoo
# E-mail: mirakuruyoo@gmail.com

import torch
import torch.nn as nn

from .BasicModule import BasicModule

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


class DPCNN(BasicModule):
    """
    DPCNN model, 3
    1. region embedding: using TetxCNN to generte
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    # max_features, opt.EMBEDDING_DIM, opt.SENT_LEN, embedding_matrix):
    def __init__(self, opt):
        super(DPCNN, self).__init__()
        self.model_name = "DPCNN"
        self.opt = opt

        if opt.USE_CHAR:
            # use char instead of word
            opt.VOCAB_SIZE = opt.CHAR_SIZE
        self.embedding = nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(opt.EMBEDDING_DIM, opt.NUM_ID_FEATURE_MAP,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP),
            nn.ReLU(),
            nn.Conv1d(opt.NUM_ID_FEATURE_MAP, opt.NUM_ID_FEATURE_MAP,
                      kernel_size=3, padding=1),
        )

        self.num_seq = opt.SENT_LEN
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

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(self.opt.BATCH_SIZE, -1)
        out = self.fc(x)
        return out
