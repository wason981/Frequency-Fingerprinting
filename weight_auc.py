# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch import argmax
import utils
import sys
sys.path.append('../../')
from model_load import load_model
from torchvision import  models
from torch import nn,load
from torch.utils.data import Dataset, DataLoader
import os
class CustomDataset(Dataset):
    def __init__(self, data_dict_list):
        self.data = data_dict_list['data']
        self.label = data_dict_list['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx])


def calculate_auc(list_a, list_b):
    l1,l2 = len(list_a),len(list_b)
    y_true,y_score = [],[]
    for i in range(l1):
        y_true.append(1)
    for i in range(l2):
        y_true.append(0)###真实标签[0,0,0,0,0,1,1,1,1,1]
    y_score.extend(list_a)
    y_score.extend(list_b)###预测分数(概率）[0.09,0.08,0.08,... 0.99,0.98,0.97]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)###设置不同的阈值，根据预测分数与不同阈值画出roc
    auc__=auc(fpr, tpr)
    return auc__

def test_similarity(models,test_loader):
    results=[]
    similarity=[]
    for model in models:
        model = model.cuda()
        model.eval()
        result=cal_pred(model,test_loader)
        results.append(result)
        model=model.cpu()
    for i in range(1,len(results)):
        simi=np.sum(results[0]==results[i])/result.shape[0]
        similarity.append(simi)
    return similarity

def cal_pred(model,dataloader):
    result=[]
    for i,(x,y) in enumerate(dataloader):
        with torch.no_grad():
            x, y = x.cuda(), y.cuda()
            model=model.cuda()
            output = model(x)
            output = output.detach()
            pred = argmax(output,dim=-1)
            result.append(pred.data.cpu().numpy())
    result=np.concatenate(result)
    return result

def auc_pickle(models_list,dataset):
    data_set = utils.load_result(dataset)
    dataset = CustomDataset(data_set)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    with  torch.no_grad():
        similarity = test_similarity(models_list, dataloader)
    #tiny——imagenet
    # list1 = similarity[:20]  # ext-p
    # list2 = similarity[20:40]  # ext-l
    # list3 = similarity[40:60]  # ext-adv
    list4 = similarity[20:25]  # prune
    # list5 = similarity[80:100]  # finetune
    # list6 = similarity[100:110]  # cifar10c ft
    # list7 = similarity[110:120]  # cifar10c irrelevant
    list8 = similarity[:20]  # irrelevant
    # print(torch.tensor(similarity).tolist())
    #cifar10
    # list1 = similarity[:20]  # ext-p
    # list2 = similarity[20:40]  # ext-l
    # list3 = similarity[40:60]  # ext-adv
    # list4 = similarity[60:75]  # prune
    # list5 = similarity[75:95]  # finetune
    # list6 = similarity[95:105]  # cifar10c ft
    # list7 = similarity[105:115]  # cifar10c irrelevant
    # list8 = similarity[115:135]  #
    # auc_p = calculate_auc(list1, list8)  ####extract-p
    # auc_l = calculate_auc(list2, list8)  ####extract-l
    # auc_adv = calculate_auc(list3, list8)  ####extract-adv
    # auc_list=[]
    # for i in range(1,100):

    return round(calculate_auc(list4, list8),2)
    # auc_prune = calculate_auc(list4, list8)  ####prune
    # auc_finetune = calculate_auc(list5, list8)  ###finetune-a
    # auc_10C = calculate_auc(list6, list7)  ####transfer-10C
    # auc_list.append(round(auc_prune, 2))
    # auc_list = [round(mean, 2), round(std, 2),
    #             round(auc_p, 2), round(auc_l, 2), round(auc_adv, 2),
    #             round(auc_prune, 2), round(auc_finetune, 2), round(auc_10C, 2)
    #             ]
    # auc_list.append(round(np.mean(auc_list[2:]), 2))
    # worksheet1.write_row('A' + str(row), auc_list)
    # row += 1
    # return round(auc_prune, 2)
    # for j in auc_list:
    #     sys.stdout.write(str("{:.2f}".format(j)))
    #     sys.stdout.write(str(' '))
    # sys.stdout.write(str('\n'))

import sys
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="AUC Calculation")
    parser.add_argument(
        "--root_path",
        type=str,
        default='/data/huangcb/WaveGAN/datasets/tiny_misclassified_distance/sample',
        help="path to images"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10','tiny_imagenet','gtsrb']
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,

    )
    parser.add_argument(
        '--version',
        type=int,
        default=4,

    )
    args = parser.parse_args()

    models_list = []
    accus = []

    #源模型 VGG16
    print('source_model')
    for i in [0]:
        globals()['source_model' + str(i)] = load_model(i, "source_model",args.dataset)
        models_list.append(globals()['source_model' + str(i)])

    # 模型提取-概率 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    # print('extract_p')
    # for i in range(20):
    #     globals()['extract_p' + str(i)] = load_model(i,"extract_p",args.dataset)
    #     models_list.append(globals()['extract_p' + str(i)])
    #
    # # #模型提取-标签 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    # print('extract_l')
    # for i in range(20):
    #     globals()['extract_l' + str(i)] = load_model(i,"extract_l",args.dataset)
    #     models_list.append(globals()['extract_l' + str(i)])
    #
    # # # #模型提取-对抗 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    # print('extract_adv')
    # for i in range(20):
    #     globals()['extract_adv' + str(i)] = load_model(i, "extract_adv",args.dataset)
    #     models_list.append(globals()['extract_adv' + str(i)])
    #
    # # # 剪枝 VGG 20
    # print('pruning')
    # for i in range(20):
    #     globals()['weight_pruning' + str(i)] = load_model(i, "weight_pruning", args.dataset)
    #     models_list.append(globals()['weight_pruning' + str(i)])
    #
    # # # #微调f-a/f-l VGG 20个
    # print('finetune')
    # for i in range(20):
    #     globals()['finetune' + str(i)] = load_model(i, 'finetune',args.dataset)
    #     models_list.append(globals()['finetune' + str(i)])
    #
    # # # #迁移学习 VGG 10个
    # print('transfer')
    # for i in range(10):
    #     globals()['transfer' + str(i)] = load_model(i, "transfer",args.dataset)
    #     models_list.append(globals()['transfer' + str(i)])
    #
    # # # #
    # # # # #迁移学习 irrelevant  VGG16 resnet18 各5个
    # print('transfer_irrelevant')
    # for i in range(10):
    #     globals()['transfer_irrelevant' + str(i)] = load_model(i, "transfer_irrelevant",args.dataset)
    #     models_list.append(globals()['transfer_irrelevant' + str(i)])

    # 无关模型 VGG13 resnet18 densenet121 mobilenet_v2 各5个
    print('irrelevant')
    for i in range(20):
        globals()['irrelevant' + str(i)] = load_model(i, "irrelevant",args.dataset)
        models_list.append(globals()['irrelevant' + str(i)])

    ## ---e-----
    # f=1
    # for e in range(1,6):
    #     e=round(0.01*e,2)
    #     p=0
    #     for i in range(10,400,10):
    #         ckpt=f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/tiny_imagenet/v9/f{f}.0/e{e}/p{p}.0/samples/{i}.pkl'
    #         auc_pickle(models_list,ckpt)

    ## ---f-----
    # p = 0
    # for f in  range(1,6):
    #     for e in range(1,2):
    #         e=round(0.01*e,2)
    #         for i in range(10,400,10):
    #             ckpt=f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/tiny_imagenet/v9/f{f}.0/e{e}/p{p}.0/samples/{i}.pkl'
    #             auc_pickle(models_list,ckpt)

    ## ---p-----
    # for f in  range(1,2):
    #     for e in range(1,2):
    #         e=round(0.01*e,2)
    #         for p in range(1, 3):
    #             for i in range(10,400,10):
    #                 ckpt=f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/tiny_imagenet/v9/f{f}.0/e{e}/p{p}.0/samples/{i}.pkl'
    #                 auc_pickle(models_list,ckpt)


    # for i in range(10):
    #     print(f'gtsrb—{i}')
    #     auc_pickle(models_list,f'/data/huangcb/WaveGAN/datasets/gtsrb_distance/sample/distance_0.{i}.pkl')
    # for i in range(10):
    #     print(f'cifar10—{i}')
    #     auc_pickle(models_list,f'/data/huangcb/WaveGAN/datasets/cifar10_distance/sample/distance_0.{i}.pkl')
    # for i in range(10):
    #     print(f'tiny_imagenet—{i}')
    #     auc_pickle(models_list,f'/data/huangcb/WaveGAN/datasets/tiny_imagenet_distance/sample/distance_0.{i}.pkl')

    w_datapath = f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/{args.dataset}/original/fregan_fingerprint.pkl'
    # m_datapath = os.path.join(args.root_path, args.dataset, "original", 'sac_m_fingerprint.pkl')
    w_list = []
    # m_list = []
    # auc_pickle(models_list,datapath)
    for ratio in range(1, 100):
        print(ratio)
        # globals()['weight_pruning' + str(i)] = load_model(i, "weight_pruning", args.dataset)
        model_weight = []
        for num in range(5):
            model = models.vgg16_bn(pretrained=False)
            in_feature = model.classifier[-1].in_features
            if args.dataset == 'cifar10':
                model.classifier[-1] = nn.Linear(in_feature, 10)
            elif args.dataset == 'tiny_imagenet':
                model.classifier[-1] = nn.Linear(in_feature, 100)
            elif args.dataset == 'gtsrb':
                model.classifier[-1] = nn.Linear(in_feature, 43)
            model.load_state_dict(load(os.path.join(
                f'/data/huangcb/WaveGAN/model/{args.dataset}/Weight_Pruning/prune_model_test_{ratio}_{num}.pth')))
            model_weight.append(model)
        models_list = models_list[0:21] + model_weight
        w_list.append(auc_pickle(models_list, w_datapath))
        # m_list.append(auc_pickle(models_list, m_datapath))
        print('fregan', w_list)
        # print('m', m_list)
        # auc_pickle(models_list, f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/tiny_imagenet/v4/1.0/0.01/samples/1900.pkl')
    # auc_pickle(models_list, '/data/huangcb/WaveGAN/projects/Freadv_GAN/output/cifar10/fregan_fingerprint.pkl')
    # auc_pickle(models_list, '/data/huangcb/WaveGAN/datasets/tiny_imagenet_misclassified_withlabel/misclassified_samples.pkl')

    # for d in range(6,7):
    #     print(d)
    #     for i in range(100,1400,100):
    #         ckpt=f'/data/huangcb/WaveGAN/projects/Freadv_GAN/output/gtsrb/v8/{round(d*0.1,1)}/0.01/samples/{i}.pkl'
    #         auc_pickle(models_list, ckpt)

