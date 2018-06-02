# coding: utf-8
# Author: Miracle Yoo, Zekang Li
import shutil
from torch.utils.data import DataLoader
from utils import *
from train import *
from config import Config
from LoadData import *
from models import LSTM, BiLSTM, TextCNN, OriTextCNN

opt = Config()

# check dir. if not existed, make dir.
folder_init()

# init PrepareData class
prep = PrepareData(opt, char=True)

# load data.
if type(opt.TRAIN_DATASET_PATH) == list:
    data_dict = prep.gen_data_dict(data_list=multi_dataset_merge(*opt.TRAIN_DATASET_PATH))
else:
    data_dict = prep.gen_data_dict(data_path=opt.TRAIN_DATASET_PATH)
test_dict = prep.gen_data_dict(opt.TEST_DATASET_PATH)

# generate vocab
if opt.IS_TRAINING:
    if opt.USE_CHAR:
        vocab_path = "./source/data/vocab_dict_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + "_char.pkl"
    else:
        vocab_path = "./source/data/vocab_dict_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + "_word.pkl"
    if os.path.exists(vocab_path):
        vocab_dict = pickle.load(open(vocab_path, "rb"))
    else:
        vocab_dict = prep.gen_vocab_dict(data_dict, test_dict)
        pickle.dump(vocab_dict, open(vocab_path, "wb"))
    title_path = "./source/data/title_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + ".pkl"
    if os.path.exists(title_path):
        title = pickle.load(open(title_path, "rb"))
    else:
        title = prep.gen_title(data_dict, test_dict)
        pickle.dump(title, open(title_path, "wb"))

else:  # opt.IS_TRAINING = False -- test
    if opt.USE_CHAR:
        vocab_path = "./source/data/vocab_dict_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + "_char.pkl"
    else:
        vocab_path = "./source/data/vocab_dict_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + "_word.pkl"
    vocab_dict = pickle.load(open(vocab_path, "rb"))
    title_path = "./source/data/title_" + opt.TRAIN_DATASET_PATH.split("/")[-1].split(".")[0] + ".pkl"
    title = pickle.load(open(title_path, "rb"))

if opt.USE_CHAR:
    opt.CHAR_SIZE = len(vocab_dict)
else:
    opt.VOCAB_SIZE = len(vocab_dict)

opt.NUM_CLASSES = len(title)
testData = prep.load_cls_data(test_dict, title, train=False)
opt.NUM_TEST = len(testData)
if opt.BANLANCE:
    trainDataSet = BalancedData(data_dict, title, opt, vocab_dict=vocab_dict)
else:
    trainData    = prep.load_cls_data(data_dict, title, train=True)
    trainDataSet = BeibeiClassification(trainData, opt, vocab_dict=vocab_dict)
testDataSet = BeibeiClassification(testData[:], opt, vocab_dict=vocab_dict)
train_loader = DataLoader(dataset=trainDataSet, batch_size=opt.BATCH_SIZE,
                          shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=True)

test_loader = DataLoader(dataset=testDataSet, batch_size=1,
                         shuffle=False, num_workers=1, drop_last=False)

if opt.IS_TRAINING:
    net = training(train_loader, test_loader, opt, title)
else:  # test
    if opt.ENSEMBLE_TEST:
        net_list = [TextCNNIncDeep.TextCNNInc, TextLSTM.TextLSTM]
        for i, model_name in enumerate(opt.MODEL_NAME_LIST):
            opt_path = opt.NET_SAVE_PATH + model_name.split(".")[0] + ".opt"
            temp_opt = pickle.load(open(opt_path, "rb"))
            net_list[i] = net_list[i](temp_opt)
        for i, _ in enumerate(net_list):
            net_list[i], *_ = net_list[i].load(opt.NET_SAVE_PATH + net_list[i].model_name + "/" + opt.MODEL_NAME_LIST[i])
        ave_test_acc, _ = ensemble_testing(test_loader, net_list, opt, title)
    else:
        net = LSTM.LSTM(opt)
        net, *_ = net.load(opt.NET_SAVE_PATH + net.model_name + "/" + opt.MODEL_NAME_LIST[0])
        print('==> Now testing model: %s ' % net.model_name)
        ave_test_loss, ave_test_acc, _ = testing(test_loader, net, opt, title)

    print('Test Acc: %.4f' % ave_test_acc)
