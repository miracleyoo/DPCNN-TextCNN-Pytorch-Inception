# coding: utf-8
# Author: Miracle Yoo

from .BasicModule import BasicModule
import torch
import torch.nn as nn
from copy import deepcopy
torch.manual_seed(1)


class TextCNNIncDeep(BasicModule):
    def __init__(self, opt):
        """
        initialize func.
        :param opt: config option class
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
        self.change_dim_conv  = nn.Conv1d(opt.TITLE_DIM*self.num_seq, opt.NUM_ID_FEATURE_MAP, kernel_size=1, stride=1)
        self.standard_pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        self.standard_batchnm = nn.BatchNorm1d(num_features=opt.NUM_ID_FEATURE_MAP)
        self.standard_act_fun = nn.ReLU()

        self.question_convs = nn.ModuleList(question_convs)
        self.fc = nn.Sequential(
            nn.Linear(opt.NUM_ID_FEATURE_MAP, opt.NUM_CLASSES),
            nn.BatchNorm1d(opt.NUM_CLASSES),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.NUM_CLASSES, opt.NUM_CLASSES)
        )

    def forward(self, question):
        question = self.encoder(question)
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        x  = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        x  = torch.cat(x, dim=1)
        xp = x
        xp = self.change_dim_conv(xp)
        x  = self.conv3x3(in_channels=x.size(1), out_channels=self.opt.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.opt.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = x+xp
        while x.size(2) > 2:
            x = self._block(x)
        x  = x.view(x.size(0), -1)
        x  = self.fc(x)
        return x

    def conv3x3(self, in_channels, out_channels, stride=1, padding=1):
        """3x3 convolution with padding"""
        _conv =  nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                         padding=padding, bias=False)
        if self.opt.USE_CUDA:
            return _conv.cuda()
        else:
            return _conv

    def _block(self, x):
        x  = self.standard_pooling(x)
        xp = x
        x  = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.opt.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x  = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.opt.NUM_ID_FEATURE_MAP)(x)
        x  = self.standard_batchnm(x)
        x  = self.standard_act_fun(x)
        x += xp
        return x


