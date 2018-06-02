# coding: utf-8
# Author: Miracle Yoo

import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable

from utils import *
import shutil
import pickle
from tqdm import tqdm
from tensorboardX import SummaryWriter
from models import TextCNNInc, TextCNNIncDeep, TextCNNIncDeep

def training(train_loader, test_loader, opt, title):
    """
    :param train_loader: train loader
    :param test_loader: test loader
    :param opt: config option class
    :param title: a list contain all of the main question
    :return: trained model
    """
    if opt.MODEL == "TextCNNInc":
        net = TextCNNInc.TextCNNInc(opt)
    elif opt.MODEL == "TextCNNIncDeep":
        net = TextCNNIncDeep.TextCNNIncDeep(opt)
    elif opt.MODEL == "TextCNNIncDeep":
        net = TextCNNIncDeep.TextCNNIncDeep(opt)

    best_acc = 0
    NUM_TRAIN = opt.BATCH_SIZE
    PRE_EPOCH = 0
    NET_PREFIX = opt.NET_SAVE_PATH + net.model_name + "/"
    print('==> Loading Model ...')
    model_name = opt.MODEL_NAME + ".pth"
    model_config = opt.MODEL_NAME + ".cfg"
    opt_save_path = opt.MODEL_NAME + ".opt"
    pickle.dump(opt, open(opt.NET_SAVE_PATH + "/" + opt_save_path, "wb"))
    if not os.path.exists(NET_PREFIX):
        os.mkdir(NET_PREFIX)
    shutil.copyfile("config.py", NET_PREFIX + model_config)
    if not os.path.exists('./source/log/' + net.model_name):
        os.mkdir('./source/log/' + net.model_name)
    if os.path.exists(NET_PREFIX + model_name) and opt.RE_TRAIN == False:
        try:
            net, PRE_STEP, best_acc = net.load(NET_PREFIX + model_name)
            print("Load existing model: %s" % (NET_PREFIX + model_name))
        except IOError:
            pass

    if opt.USE_CUDA: net.cuda()

    criterion = nn.CrossEntropyLoss()
    if opt.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)
    elif opt.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.LEARNING_RATE)
    elif opt.OPTIMIZER == 'RMSP':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.LEARNING_RATE)
    else:
        raise NameError("This optimizer isn't defined")

    writer = SummaryWriter(opt.SUMMARY_PATH)

    # Start training
    print("Now Tensorboard running. The summary directory is %s" % opt.SUMMARY_PATH)
    for step, data in enumerate(train_loader):
        train_loss = 0
        train_acc = 0
        net.train()
        inputs, labels, sent = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Do statistics for training
        train_loss += loss.data[0]
        _, predicts = torch.max(outputs, 1)
        num_correct = (predicts == labels).sum()
        train_acc += num_correct.data[0]

        writer.add_scalar("Train/loss", train_loss / NUM_TRAIN, step+PRE_STEP)
        writer.add_scalar("Train/acc", float(train_acc) / NUM_TRAIN, step+PRE_STEP)

        # testing
        if step % opt.TEST_STEP == 0 and step != 0:
            test_loss, test_acc, topnacc = testing(test_loader, net, opt, title)
            writer.add_scalar("Test/loss", test_loss, step+PRE_STEP)
            writer.add_scalar("Test/acc", test_acc, step+PRE_STEP)

            if test_acc > best_acc:
                best_acc = test_acc
                net.save((step + PRE_EPOCH), best_acc, model_name)

    print('==> Training Finished. Current model is %s. The highest test acc is %.4f' % (net.model_name, best_acc))
    return net


def testing(test_loader, net, opt, title):
    """
    :param test_loader: test loader
    :param net: trained net
    :param opt: config option class
    :param title: a list contain all of the main question
    :return: test loss, test acc, top N acc(list)
    """
    NUM_TEST = opt.NUM_TEST
    test_loss = 0
    test_acc = 0
    topn_acc = [0] * opt.TOP_NUM
    equal_pairs = get_equal_pairs(title)
    criterion = nn.CrossEntropyLoss()
    if opt.USE_CUDA: net.cuda()

    net.eval()
    for i, data in enumerate(test_loader):
        inputs, labels, sent = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        _, predicts = torch.max(outputs, 1)
        if opt.USE_CUDA:
            labels = labels.cpu().data
            predicts = predicts.cpu().data
        else:
            labels = labels.data
            predicts = predicts.data

        if opt.USE_PAIR_MAPPING:
            num_correct = use_pairs_mapping(predicts.tolist(), labels.tolist(), equal_pairs)
        else:
            num_correct = (predicts == labels).sum()

        for i in range(opt.TOP_NUM):
            predictsn = np.array(outputs.data.sort(descending=True, dim=1)[1])[:, :(i + 1)]
            if opt.USE_PAIR_MAPPING:
                topn_acc[i] += use_pairs_mapping(predictsn.tolist(), labels.tolist(), equal_pairs)
            else:
                for j in range(len(labels)):
                    if labels[j] in predictsn[j]:
                        topn_acc[i] += 1

        # Do statistics for training
        test_loss += loss.data[0]
        test_acc += num_correct

    test_loss = float(test_loss) / NUM_TEST
    test_acc = float(test_acc) / NUM_TEST
    topn_acc = [float(x) / NUM_TEST for x in topn_acc]
    return test_loss, test_acc, topn_acc


def ensemble_testing(test_loader, net_list, opt, title):
    """
    :param test_loader: test loader
    :param net_list: the list which need to be used in ensemble learning.
    :param opt: config option class
    :param title: a list contain all of the main question
    :return: test loss, test acc, top N acc(list)
    """
    NUM_TEST = opt.NUM_TEST
    theta_list = opt.MODEL_THETA_LIST
    test_acc = 0
    topn_acc = [0] * opt.TOP_NUM
    equal_pairs = get_equal_pairs(title)
    if opt.USE_CUDA: net_list = [net.cuda() for net in net_list]
    for i, _ in enumerate(net_list):
        net_list[i].eval()
    for i, data in enumerate(test_loader):
        inputs, labels, sent = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        if opt.USE_CUDA:
            outputs = Variable(torch.zeros(1, opt.NUM_CLASSES)).cuda()
        else:
            outputs = Variable(torch.zeros(1, opt.NUM_CLASSES))
        for i in range(len(net_list)):
            net = net_list[i]
            theta = theta_list[i]
            output = net(inputs)
            output = nn.functional.softmax(output)
            outputs += theta * output
        _, predicts = torch.max(outputs, 1)

        if opt.USE_CUDA:
            labels = labels.cpu().data
            predicts = predicts.cpu().data
        else:
            labels = labels.data
            predicts = predicts.data

        if opt.USE_PAIR_MAPPING:
            num_correct = use_pairs_mapping(predicts.tolist(), labels.tolist(), equal_pairs)
        else:
            num_correct = (predicts == labels).sum().data[0]

        for i in range(opt.TOP_NUM):
            predictsn = np.array(outputs.data.sort(descending=True, dim=1)[1])[:, :(i + 1)]
            if opt.USE_PAIR_MAPPING:
                topn_acc[i] += use_pairs_mapping(predictsn.tolist(), labels.tolist(), equal_pairs)
            else:
                for j in range(len(labels)):
                    if labels[j] in predictsn[j]:
                        topn_acc[i] += 1

        # Do statistics for training
        test_acc += num_correct

    test_acc = float(test_acc)
    return test_acc / NUM_TEST, topn_acc
