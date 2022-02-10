# Conduct multi-teacher distillation
# 先写个30轮的预热。 get global model_init
# Next round:
# select again
# Load from global/local_model_dic2
# Train local model with unlabeled data:forward的同时产生 psudo label vector,（判断置信度），算损失，反向传播。
# 剪枝
# save to w_locals.
# Distill(w_locals) ：Generate 5 local models,dataloader:labeled dataset,forward(x),re-define loss function,backward.
# Save to local_model_dic2
import os
import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
import numpy as np
# np.random.seed(0)
import copy
import gc
import matplotlib.pyplot as plt
# from tqdm import tqdm
import random

random.seed(0)
# from torchviz import make_dot
# from sklearn import metrics
from torch.autograd import Variable
import itertools
import logging
import os.path
from PIL import Image
from torch.utils.data.sampler import Sampler
from models.byol import BYOL, BYOLP
import re
import argparse
import shutil
import time
import math
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset
from torch import nn, autograd
from tools.prune_utils import recover_network, prune_network
from tools.fed import FedAvg_nonzero
from configs import get_args
from augmentations import get_aug
from models import get_model
from models.backbones import *
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from tools.test import test_img, test_img2
from tools.loss import softmax_mse_loss, softmax_kl_loss, symmetric_mse_loss, softmax_kl_loss2
from tools.ramps import get_current_consistency_weight
from datasets.sampling import iid, noniid, DatasetSplit, iid2, noniid2
from tools.fed import FedAvg

from tools.prune_utils import *

from tools.prune_utils import updateBN, prune_globel_simple_weight, prune_network_sliming, recover_network, \
    zero_out_gradient
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary


def ablation_fed_avg(model_locals, cfg_mask_locals):
    avg_models = []
    w_locals_filled_zeros = []
    for model, cfg_local_mask in zip(model_locals, cfg_mask_locals):
        model.backbone = recover_network(model.backbone, cfg_local_mask, args.backbone, dataset=args.dataset, args=args)
        w_locals_filled_zeros.append(copy.deepcopy(model.backbone.state_dict()))
    w_locals_avg = FedAvg_nonzero(w_locals_filled_zeros)
    for w, cfg_local_mask in zip(w_locals_avg, cfg_mask_locals):
        model_local_pruned = prune_network(w, cfg_local_mask, args.backbone, args.dataset)
        model_local_sum = sum(p.numel() for p in model_local_pruned.teacher.parameters())
        print("parameters:",model_local_sum)

        avg_models.append(model_local_pruned)
    return avg_models  # 返回的backbone list


def main(device, args):
    log_dir = args.log_dir
    log_fn = args.log_fn
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")
    stderr = sys.stderr
    # sys.stderr=log_fp

    # define loss function
    loss1_func = nn.CrossEntropyLoss()
    loss2_func = softmax_kl_loss
    loss3_func = softmax_kl_loss2
    # define dataset
    dataset_kwargs = {
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'download': True,
        'debug_subset_size': args.batch_size if args.debug else None
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs = {
        'batch_size': args.batch_size,  # *5,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataloader_unlabeled_kwargs2 = {
        'batch_size': 1,  # *5,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    dataset_train = get_dataset(
        transform=get_aug(args.dataset, True),
        train=True,
        **dataset_kwargs
    )
    # dataset_train2 = get_dataset(
    #     transform=ToTensor()
    #     train=True,
    #     **dataset_kwargs),
    # Sample
    if args.iid == 'iid':
        dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_train, dict_users_unlabeled_test = iid2(
            dataset_train, args.num_users, args.label_rate, args.seed)
    else:
        dict_users_labeled, dict_users_unlabeled, dict_users_unlabeled_train, dict_users_unlabeled_test = noniid2(
            dataset_train, args.num_users, args.label_rate, args.seed)

        # dict_users_labeled2, dict_users_unlabeled2 = noniid(dataset_train2, args.num_users, args.label_rate)
    # print("labeled:",dict_users_labeled)
    # for i in range(args.num_users):
    #     print("------------------------")
    #     print("unlabeled:",i)
    #     print(dict_users_unlabeled[i])
    # k=list(dict_users_unlabeled[0])
    # print("k:",k)
    # print("sample",dataset_train[k[1]][0])
    # ((images1, images2), labels)=dataset_train[k[1]]
    # images1 = images1.unsqueeze(0)
    # print("images1:",images1.size())
    # print("images2:", images1.size())
    # print("labels:",labels)

    # initialize global model
    # model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)
    if (args.heat == 0):  ##不预热
        if (args.backbone == "Vgg_backbone"):
            if (args.dataset == "cifar10"):
                model_glob_heat = torch.load('Model_Cifar_vgg19.pkl', map_location='cpu')
            if (args.dataset == "svhn"):
                model_glob_heat = torch.load('Model_Svhn_vgg19.pkl', map_location='cpu')
        else:
            model_glob_heat = torch.load('Model_Mnist_vgg19.pkl', map_location='cpu')
        model_glob_sum = sum(p.numel() for p in model_glob_heat.parameters())
        model_glob_heat.to(device)
    else:
        if (args.backbone == "Vgg_backbone"):
            model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)
        if (args.backbone == "vgg"):
            model_glob = vgg().to(device)
        if (args.backbone == "Mnist"):
            model_glob = get_model('global', args.backbone, dataset=args.dataset).to(device)

        # model_glob=VGG2().to(device)
        model_glob_sum = sum(p.numel() for p in model_glob.parameters())
        # print("parameters")
        # print(list(model_glob.parameters()))
        # summary(model_glob, (3, 32, 32))
        print("parameters number of vgg:", model_glob_sum)
        # if torch.cuda.device_count() > 1: model_glob = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_glob)

        # Initialize indexs/accuracy/model list......
        model_local_idx = set()  # local model index
        model_local_dict_backbone = {}
        model_local_dict_fc = {}
        accuracy = []
        best_test_acc = float('-inf')
        best_train_acc = float('-inf')
        lr_scheduler = {}
        accuracy_log = []
        # optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.1)
        # preheating
        print("--------------------warm up-------------------------", args.heat_epochs)
        for iter in range(args.heat_epochs):
            model_glob.train()
            optimizer = torch.optim.SGD(model_glob.parameters(), lr=0.1)
            # Load labeled data :batch size is setted here
            train_loader_labeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
                shuffle=True,
                **dataloader_kwargs
            )
            for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
                if torch.cuda.is_available():
                    labels = labels.to(device)
                    images1 = images1.to(device)

                    # images2=images2.cuda()
                # model_glob.zero_grad()##optimizor??
                # print("image_size",images1.size())
                z1 = model_glob(images1)
                # Do not consider consistency loss here
                loss = loss1_func(z1, labels)
                # print("loss:",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print("backbone:",model_glob.state_dict())
            # del train_loader_labeled  # What is del???
            # gc.collect()
            # torch.cuda.empty_cache()
            # test model
            if iter % 1 == 0:
                dataset_kwargs['download'] = False
                # Load test set
                test_loader = torch.utils.data.DataLoader(
                    dataset=get_dataset(
                        transform=get_aug(args.dataset, False, train_classifier=False),
                        train=False,
                        **dataset_kwargs),
                    shuffle=False,
                    **dataloader_kwargs
                )
                model_glob.eval()
                acc, loss_train_test_labeled = test_img(model_glob, test_loader, args)
                if acc > best_test_acc:
                    best_test_acc = acc
                accuracy.append(str(acc))
                if iter % 1 == 0:
                    print('Round {:3d}, Best Test Acc {:.2f}%'.format(iter, acc))

                del test_loader
                gc.collect()
                torch.cuda.empty_cache()
            print("Warm up stage:accuracy:", best_test_acc, file=log_fp, flush=True)
        if (args.dataset == "svhn"):
            print("save model as Model_Svhn_vgg19.pkl")
            torch.save(model_glob, 'Model_Svhn_vgg19.pkl')
        if (args.dataset == "cifar10"):
            print("save model as Model_Cifar_vgg19.pkl")
            torch.save(model_glob, 'Model_Cifar_vgg19.pkl')
        if (args.dataset == 'mnist'):
            print("save model as Model_Mnist_vgg19.pkl")
            torch.save(model_glob, 'Model_Mnist_vgg19.pkl')

        model_glob_heat = model_glob
        model_glob_sum = sum(p.numel() for p in model_glob_heat.parameters())

    # model_glob_heat=torch.load('Model_svhn_vgg19.pkl',map_location='cpu')
    # model_glob_heat=model_glob
    # model_glob_sum = sum(p.numel() for p in model_glob_heat.parameters())

    model_glob_heat.to(device)
    model_list = [[] for idx in range(args.num_users)]
    accuracy_list = [[] for idx in range(args.num_users)]
    averge_accuracy_list = []
    model_pruned_statisic = np.zeros(args.num_users, dtype=int)
    print("lengh of model list:", model_list)
    # test each clients' model
    for idx in range(args.num_users):
        test_loader_unlabeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),  # load unlabeled data for user i
            shuffle=True,
            **dataloader_unlabeled_kwargs
        )
        model_glob_heat.eval()
        with torch.no_grad():
            acc, loss_train_test_labeled = test_img2(model_glob_heat, test_loader_unlabeled, args)
        accuracy_list[idx] = acc.numpy()
    print("accuracy_list", accuracy_list)
    total_num_list = []
    total_num_list.append(model_glob_sum * 10)
    # Training
    print("==============begin training======================")
    for iter in range(args.num_epochs):
        id_accuracy_list1 = []  # idxs before training
        id_accuracy_list2 = []  # ...  before distillationprint("training iter:",iter,"of",args.num_epochs)
        id_accuracy_list3 = []  # ... after distillationprint("training iter:", iter, "of", args.num_epochs,file=log_fp, flush=True)
        # print("accuracy", accuracy_list)
        # averge_accuracy= sum(accuracy_list) / len(accuracy_list)
        # print("average accuracy", averge_accuracy)
        # print("average accuracy", averge_accuracy,file=log_fp, flush=True)
        # averge_accuracy_list.append(averge_accuracy)
        # print("average accuracy list", averge_accuracy_list, file=log_fp, flush=True)

        # if iter%10==0:
        #     print("average accuracy list", averge_accuracy_list)

        # print("-------------------------------------------------------------------------------------------------")
        # preparation
        w_locals, loss_locals, loss0_locals, loss2_locals, id_list = [], [], [], [], []
        model_list_temp = []
        cfg_mask_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), 10, replace=False)
        # Begin training
        total_num = 0
        # total_num_list=[]
        for idx in idxs_users:
            # model_local = BYOLP(model_glob_heat.backbone).to(device) #initialize local model
            # 读取global_head model或者已经训练过的local model(unfinished) 不过模型大小变了。。
            print("===Load model===")
            if model_list[idx]:
                # print("{:3d} idx has been trained".format(idx))
                model_local = model_list[idx]  # Download personalized model
            else:
                # print("{:3d} idx has not been trained".format(idx))
                model_local = copy.deepcopy(model_glob_heat)
                # model_local = BYOLP(model_glob_heat.backbone).to(device)
                # model_local.backbone.load_state_dict(model_glob_heat.backbone.state_dict())#Download global model
                # model_local.fc.load_state_dict(model_glob_heat.fc.state_dict())#Download global model

            # 读取encoder
            # model_local.target_encoder.load_state_dict(model_local.online_encoder.state_dict())
            # model_local_idx = model_local_idx | set([idx])  # 参与过训练的id列表
            # Load unlabel dataset
            train_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_train[idx]),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            test_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local.eval()
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            print("-------before training  accuracy of:", idx)
            print("Accuracy:", acc)
            id_accuracy_list1.append(acc)
            # accuracy_list[idx]=acc.numpy()
            dict_unlabeled_highscore = highscoresampling(dataset_train, dict_users_unlabeled_train[idx], args.threhold,
                                                         model_local, device)
            # print("unlabeled dataset", dict_users_unlabeled[idx])
            print("unlabeled dataset", len(dict_users_unlabeled_train[idx]))
            print("fine_tuned dataset:", len(dict_unlabeled_highscore))
            # if len(dict_unlabeled_highscore) == 0:
            dict_unlabeled_highscore = dict_users_unlabeled_train[idx]
            model_label = copy.deepcopy(model_local)
            # optimizer = get_optimizer(
            #     args.optimizer, model_local,
            #     lr=args.base_lr * args.batch_size / 256,
            #     momentum=args.momentum,
            #     weight_decay=args.weight_decay)
            optimizer = torch.optim.SGD(model_local.parameters(), lr=0.01)
            model_local.train()  # Begin to train local model
            # Train local data
            print("{:3d} begin trainning".format(idx))

            # with torch.no_grad():
            #     acc, loss_train_test_labeled = test_img2(model_local, train_loader_unlabeled, args)
            # print("before training  accuracy of:",idx)
            # print("Accuracy:",acc)
            # accuracy_list[idx]=acc.numpy()
            # print("accuracy_list",accuracy_list)
            ###save  local model
            for j in range(args.local_ep):
                for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled):
                    labels = labels.to(device)
                    images1 = images1.to(device)
                    # model_local.zero_grad()
                    # loss和反向传播 需要再设计下 还有batch size
                    z1 = model_local(images1.to(device, non_blocking=True))
                    with torch.no_grad():
                        label1 = model_label(images1)
                        label1_hard = label1.argmax(dim=1)
                        # label2_hard=label2.argmax(dim=1)
                    # label1_soft=softmax(label1)
                    loss = loss1_func(z1, label1_hard)
                    # print(idx,"loss ",loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # Save local model need modified
                # w_locals.append(copy.deepcopy(model_local.backbone.state_dict()))
                # model_local_dict_[idx] = [model_local.projector.state_dict(), model_local.target_encoder.state_dict()]
            # with torch.no_grad():
            #     acc, loss_train_test_labeled = test_img2(model_local, train_loader_unlabeled, args)
            # print("before prune accuracy of:", idx)
            # print("Accuracy:", acc)
            # model_local.eval()
            # with torch.no_grad():
            #     acc, loss_train_test_labeled = test_img2(model_local, train_loader_unlabeled, args)
            # print("------before prunning accuracy of:", idx)
            # print("Accuracy:", acc)
            ###Finetune np.argmax(model(data[rnd:rnd+1]).numpy()

            print("begin prune")
            # model_local_sum = sum(p.numel() for p in model_local.teacher.parameters())
            model_local_sum = sum(p.numel() for p in model_local.parameters())
            print("local parameters before training:", model_local_sum)
            ###剪枝
            if model_pruned_statisic[idx] < args.prunetimes:
                if (args.backbone == "vgg"):
                    model_classifier = copy.deepcopy(model_local.classifier)
                    model_local, cfg_mask = prune_network_sliming(model_local, args.prunerate, args.backbone,
                                                                  args.dataset, args.device)
                    model_local.classifier = model_classifier
                else:
                    model_local.backbone, cfg_mask = prune_network_sliming(model_local.backbone, args.prunerate,
                                                                           args.backbone,args.dataset, args.device)
                    model_local.teacher = nn.Sequential(model_local.backbone, model_local.fc)
                    model_local_sum = sum(p.numel() for p in model_local.teacher.parameters())
                    model_pruned_statisic[idx] += 1

                cfg_mask_locals.append(cfg_mask)
                print("End prune~~~times of been pruned", model_pruned_statisic[idx])  # 咋传的参数这么多
                print("local parameters after prunning:", model_local_sum)

            else:
                model_pruned_statisic[idx] += 1
                print("Times of been trained", model_pruned_statisic[idx])
                print("Don not prune")
            ###剪枝
            # model_local.eval()
            # with torch.no_grad():
            #     acc, loss_train_test_labeled = test_img2(model_local, train_loader_unlabeled, args)
            # print("------after prunning accuracy of:",idx)
            # print("Accuracy:",acc)
            ##Finetune np.argmax(model(data[rnd:rnd+1]).numpy()
            print("begin finetune")
            ##Generate a new dataset with high performance

            optimizer = torch.optim.SGD(model_local.parameters(), lr=0.01)
            train_loader_unlabeled_finetune = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_unlabeled_highscore),  # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local.train()
            for j in range(args.local_finetune):
                for i, ((images1, images2), labels) in enumerate(train_loader_unlabeled_finetune):
                    labels = labels.to(device)
                    images1 = images1.to(device)
                    # images1 =images1.unsqueeze(0)
                    # model_local.zero_grad()
                    z1 = model_local(images1)
                    with torch.no_grad():
                        label1 = model_label(images1)
                        label1_hard = label1.argmax(dim=1)
                        # print("label:",label1_hard)
                        # print("Real label:",labels)
                        # # label2_hard = label2.argmax(dim=1)
                        # label1_soft=softmax(label1)
                        # # print("label1_soft:",label1_soft)
                        # label1_Relu=F.relu(label1)+1
                        # print("RELU:",label1_Relu)
                        # confidence_score=label1_Relu.max(dim=1)/sum(label1_Relu,dim=1)#可能要改设备
                        # print("confidence score", confidence_score)
                    optimizer.zero_grad()
                    loss = loss1_func(z1, label1_hard)
                    loss.backward()
                    optimizer.step()

            ###Finetune
            ##test finetuned model
            dataset_kwargs['download'] = False

            # test_loader = torch.utils.data.DataLoader(
            #     dataset=get_dataset(
            #         transform=get_aug(args.dataset, False, train_classifier=False),
            #         train=False,
            #         **dataset_kwargs),
            #     shuffle=False,
            #     **dataloader_kwargs
            # )
            model_local.eval()
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            id_accuracy_list2.append(acc)
            print("Accuracy after finetune", acc)
            ###save  local model
            # w_locals.append(copy.deepcopy(model_local.backbone.state_dict()))
            # w_locals.append(copy.deepcopy(model_local))
            # total_num =total_num +sum(p.numel() for p in model_local.teacher.parameters())
            total_num = total_num + sum(p.numel() for p in model_local.parameters())

            id_list.append(idx)

            model_list_temp.append(copy.deepcopy(model_local))
            # model_local_dict_backbone[idx] = [copy.deepcopy(model_local.backbone.state_dict())]
            # model_local_dict_fc[idx]=[copy.deepcopy(model_local.fc.state_dict())]
            del model_local
            gc.collect()
            del model_label
            gc.collect()
            del train_loader_unlabeled
            gc.collect()
            torch.cuda.empty_cache()
        ###Multiteacher+finetune  model_glob.backbone.load_state_dict(w_glob)###How to load stored model parameters.
        # train loader
        train_loader_labeled = torch.utils.data.DataLoader(
            dataset=DatasetSplit(dataset_train, dict_users_labeled),  # load labeled data from dataset_train
            shuffle=True,
            **dataloader_kwargs
        )
        total_num_list.append(total_num)
        print("total_num_list", total_num_list, file=log_fp, flush=True)
        print("total_num_list", total_num_list)
        # num_locals = len(w_locals)
        num_locals = 10
        print("agreegation:")
        model_backbone_list = ablation_fed_avg(model_list_temp, cfg_mask_locals)
        count = 0
        for backbone in model_backbone_list:
            model_local=model_list_temp[count]
            model_local.teacher = nn.Sequential(backbone, model_local.fc)
            model_list[id_list[count]]=copy.deepcopy(model_local)
            del  model_local
            gc.collect()
            torch.cuda.empty_cache()
            count += 1

        # global finetune
        for idx in idxs_users:
            for iter in range(args.distill_round):
                model=model_list[idx]
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                # Load labeled data :batch size is setted here
                for batch_idx, ((images1, images2), labels) in enumerate(train_loader_labeled):
                    if torch.cuda.is_available():
                        labels = labels.to(device)
                        images1 = images1.to(device)

                        # images2=images2.cuda()
                    # model_glob.zero_grad()##optimizor??
                    # print("image_size",images1.size())
                    z1 = model(images1)
                    # Do not consider consistency loss here
                    loss = loss1_func(z1, labels)
                    # print("loss:",loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        for idx in idxs_users:
            test_loader_unlabeled = torch.utils.data.DataLoader(
                dataset=DatasetSplit(dataset_train, dict_users_unlabeled_test[idx]),
                # load unlabeled data for user i
                shuffle=True,
                **dataloader_unlabeled_kwargs
            )
            model_local = model_list[idx]
            with torch.no_grad():
                acc, loss_train_test_labeled = test_img2(model_local, test_loader_unlabeled, args)
            accuracy_list[idx] = acc.numpy()
            id_accuracy_list3.append(acc)

        print("===============================================", file=log_fp, flush=True)
        print("round:", iter, file=log_fp, flush=True)
        print("round:", iter)
        print("Participants:", idxs_users, file=log_fp, flush=True)
        print("Accuracy before training:", file=log_fp, flush=True)
        print(id_accuracy_list1, file=log_fp, flush=True)
        print(id_accuracy_list1)
        print("Accuracy After finetune:", file=log_fp, flush=True)
        print(id_accuracy_list2, file=log_fp, flush=True)
        print(id_accuracy_list2)
        print("Accuracy After distillation:", file=log_fp, flush=True)
        print(id_accuracy_list3, file=log_fp, flush=True)
        print(id_accuracy_list3)
        print("=============================================", file=log_fp, flush=True)

        print("accuracy", accuracy_list)
        averge_accuracy = sum(accuracy_list) / len(accuracy_list)
        print("accuracy", accuracy_list)
        print("accuracy", accuracy_list, file=log_fp, flush=True)
        print("average accuracy", averge_accuracy)
        print("average accuracy", averge_accuracy, file=log_fp, flush=True)
        averge_accuracy_list.append(averge_accuracy)
        print("average accuracy list", averge_accuracy_list, file=log_fp, flush=True)
        print("-------------------------------------------", file=log_fp, flush=True)

        # print("w_locals:",w_locals[1])
        # for model_temp in w_locals:
        #     del model_temp
        #     gc.collect()
        del train_loader_labeled
        gc.collect()
        torch.cuda.empty_cache()
        # print("w_locals:",w_locals[1])
        ###Multiteacher+finetune
    print("accuracy trace:", averge_accuracy_list)
    print("total_num_list", total_num_list)
    plt.xlabel("Training Round")
    plt.ylabel("Average Accuracy")
    plt.plot(averge_accuracy_list)
    plt.savefig('cifar10(non-iid).svg')
    plt.show()
    plt.xlabel("Training Round")
    plt.ylabel("numbers of parameters")
    plt.plot(total_num_list)
    plt.savefig('Traffic-Cifar10(noniid).svg')
    plt.show()
    log_fp.close()
    sys.stderr = stderr


def highscoresampling(dataset_train, dict_users_unlabeled, threhold, model, device):
    k = list(dict_users_unlabeled)
    dict_unlabeled_highscore = set()
    num_classes = 10
    for image_id in k:
        ((images1, images2), labels) = dataset_train[image_id]
        # labels = labels.to(device)
        images1 = images1.unsqueeze(0)
        images1 = images1.to(device)
        with torch.no_grad():
            # model_local.zero_grad()
            psudo_label = model(images1)

            psudo_label_hard = psudo_label.argmax(dim=1)
            psudo_label_soft = softmax(psudo_label)
            confidence_score = psudo_label_soft.max()
            confidence_score_np = confidence_score.cpu().numpy()
            evidence = F.relu(psudo_label)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

            # print("probalbility:",prob)
            uncertainty_np = uncertainty.cpu().numpy()
            # print("uncertainty:",uncertainty.flatten())
            # print("confidence_score_np:",confidence_score_np)
            # print("softmax:",psudo_label_soft)
            # print("confidence score:",confidence_score_np)
        if uncertainty_np < 0.7:
            # print("threhold",threhold)
            # print("yes",image_id)
            dict_unlabeled_highscore.add(image_id)
        # else:
        #     print("no")
    return dict_unlabeled_highscore


def Multiteacher(w_locals, id_list, model_list, data_loader, args, device):
    ###begin distillation
    # z1_list=[]
    # z2_list=[]
    loss1_func = nn.CrossEntropyLoss()
    loss3_func = softmax_kl_loss2
    loss2_func = softmax_kl_loss
    num_locals = len(w_locals)
    for k in range(args.distill_round):
        for batch_idx, ((images1, images2), labels) in enumerate(data_loader):
            labels = labels.to(device)
            images1 = images1.to(device)
            # generate logits
            z1_list = []
            z2_list = []
            sum_z1 = 0
            sum_z2 = 0
            for model in w_locals:
                model.eval()
                with torch.no_grad():
                    z1 = model(images1)
                    z1_list.append(softmax(z1, 2))
                    sum_z1 += softmax(z1)
                    # z2_list.append(softmax(z2))
                    # sum_z2 += softmax(z2)
            temp = 0
            for model in w_locals:
                with torch.no_grad():
                    teacher_logits_1 = (sum_z1 - z1_list[temp]) / (num_locals - 1)
                    # print("temp:",temp)
                    # print("teacher logits",teacher_logits_1)
                    # teacher_logits_2=(sum_z2-z2_list[temp])/(num_locals-1)
                model.train()

                optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
                pred = model(images1)
                loss1 = loss1_func(pred, labels)  # 0.34
                # print("loss1",loss1)
                loss2 = loss2_func(softmax(pred, 2), teacher_logits_1)  ##不知道匹不匹配 0.0 10.33
                # print("loss_multi",loss2)
                loss = loss1 + loss2
                # print("loss_all",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                temp += 1
    ##fine-tune
    ##更新模型
    temp = 0
    for model in w_locals:
        # print("temp",temp)
        id = id_list[temp]
        c = model_list[id]
        del c
        gc.collect()
        # torch.cuda.empty_cache()
        model_list[id] = model
        temp += 1
    torch.cuda.empty_cache()
    print("length of model list:", len(model_list))


def softmax(X, T=1):
    X = X / T
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)