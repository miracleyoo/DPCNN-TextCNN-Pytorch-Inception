# coding: utf-8
# Author: Miracle Yoo
import torch


class Config(object):
    def __init__(self):
        # 公共参数设置
        self.USE_CUDA           = torch.cuda.is_available()
        self.MODEL              = "TextCNNIncDeep"
        self.MODEL_NAME         = "TextCNNIncDeep_CHAR_v9"
        self.RUNNING_ON_SERVER  = True
        self.SUMMARY_PATH       = "summary/TextCNNIncDeep_CH_AR_v9"
        self.NET_SAVE_PATH      = "./source/trained_net/"
        self.TRAIN_DATASET_PATH = "/disk/Beibei_Dataset/train_data_v5.csv"
        self.TEST_DATASET_PATH  = "/disk/Beibei_Dataset/yibot_test_1617.csv"
        self.NUM_EPOCHS         = 50000
        self.BATCH_SIZE         = 400
        self.NUM_TRAIN          = self.NUM_EPOCHS * self.BATCH_SIZE
        self.NUM_TEST           = 0
        self.TEST_STEP          = 100
        self.TOP_NUM            = 3
        self.NUM_WORKERS        = 8
        self.IS_TRAINING        = True
        self.ENSEMBLE_TEST      = False
        self.LEARNING_RATE      = 0.001
        self.RE_TRAIN           = False
        self.USE_PAIR_MAPPING   = False
        self.USE_TRAD2SIMP      = False
        self.TEST_POSITION      = 'Gangge Server'

        # 模型共享参数设置
        self.OPTIMIZER          = 'Adam'
        self.USE_CHAR           = True
        self.SENT_LEN           = 30
        self.USE_WORD2VEC       = False
        self.BANLANCE           = True
        self.NUM_CLASSES        = 1890
        self.EMBEDDING_DIM      = 300
        self.VOCAB_SIZE         = 20029
        self.CHAR_SIZE          = 3403

        # LSTM模型设置
        self.LSTM_HID_SIZE      = 512
        self.LSTM_LAYER_NUM     = 2
        self.K_MAX_POOLING      = 1

        # TextCNN模型设置
        self.TITLE_DIM          = 512
        self.LINER_HID_SIZE     = 1409
        self.KERNEL_SIZE        = [2, 3, 4, 5]
        self.TITLE_EMBEDDING    = 256   # 仅用在TextCNNCos中

        # DilaTextCNN模型设置
        self.DILA_TITLE_DIM = 20

        # TextCNNInc模型设置
        self.SIN_KER_SIZE = [1, 3]  # single convolution kernel
        self.DOU_KER_SIZE = [(1, 3), (3, 5)]  # double convolution kernel

        # TextCNNIncDeep模型设置
        self.NUM_ID_FEATURE_MAP = 250

        # 模型融合
        self.MODEL_NAME_LIST    = ["TextCNNINC_CHAR_v5.pth", "TextLSTM_CHAR_v5.pth"]
        self.MODEL_THETA_LIST   = [1, 0.8]

