# coding: utf-8
# Author: Miracle Yoo

from .BasicModule import BasicModule
import torch
import torch.nn as nn
from copy import deepcopy
torch.manual_seed(1)

class TextCNNInc(BasicModule):
    def __init__(self, opt):
        """
        initialize func.
        :param opt: config option class
        """
        super(TextCNNInc, self).__init__()
        self.model_name = "TextCNNInc"

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

        question_convs = deepcopy(question_convs1)
        question_convs.extend(question_convs2)
        self.question_convs = nn.ModuleList(question_convs)
        self.num_seq = len(opt.DOU_KER_SIZE)+len(opt.SIN_KER_SIZE)
        self.fc = nn.Sequential(
            nn.Linear(self.num_seq * opt.TITLE_DIM, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES)
        )

    def forward(self, question):
        question = self.encoder(question)
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        question_out = [question_conv(question.permute(0, 2, 1)) for question_conv in self.question_convs]
        conv_out = torch.cat(question_out, dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits

