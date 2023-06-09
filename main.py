import os.path

import numpy as np
from sklearn.metrics import roc_auc_score

from Dataloader import Dataloader

import argparse
import platform
import torch
from torch.nn.init import xavier_uniform_
from ganf import Ganf
from models.GANF import GANF
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_f1_scores(lables, scores, abnormal_points_number=None):
    '''
    :param lables: ground truth labels
    :param scores: -log probability score, greater means more abnormal
    :param abnormal_points_number: number of abnormal points in the dataset, None means calculate it automatically.
    :return:
    '''

    if abnormal_points_number is None:
        abnormal_points_number = int(np.sum(lables))
    abnormal_ranking = np.argsort(scores)[::-1]
    abnormal_points = abnormal_ranking[:abnormal_points_number]
    predicted_labels = np.zeros(lables.shape, dtype=float)
    predicted_labels[abnormal_points] = 1.
    print(lables)
    print(predicted_labels)
    tp = np.where((predicted_labels == 1) & (lables == 1), 1., 0.).sum()
    fp = np.where((predicted_labels == 1) & (lables == 0), 1., 0.).sum()
    tn = np.where((predicted_labels == 0) & (lables == 0), 1., 0.).sum()
    fn = np.where((predicted_labels == 0) & (lables == 1), 1., 0.).sum()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return recall,precision,f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--auto_anomaly_ratio', action='store_true')
    parser.add_argument('-B', '--batch_norm', action='store_true')
    parser.add_argument('-D', '--save_dir', default='save/tmp', type=str)
    parser.add_argument('-G', '--cuda', action='store_true')
    parser.add_argument('-N', '--normalize', action='store_true')
    parser.add_argument('-T', '--test_only', action='store_true')
    parser.add_argument('-a', '--anomaly_ratio', default=0.1, type=float,
                        help='smd: 0.05, swat: 0.1214, wadi.old: 0.0384, wadi.new: 0.0577')
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-d', '--dataset', default='swat', type=str)
    parser.add_argument('-e', '--epoch', default=50, type=int)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    parser.add_argument('-i', '--hidden_size', type=int, default=32,
                        help='Hidden layer size for MADE (and each MADE block in an MAF).')
    parser.add_argument('-k', '--n_blocks', type=int, default=1,
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('-l', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-r', '--train_set_ratio', default=0.6, type=float)
    parser.add_argument('-s', '--stride', default=10, type=int)
    parser.add_argument('-w', '--window_size', default=60, type=int)
    parser.add_argument('-y', '--hidden_layers', type=int, default=1, help='Number of hidden layers in each MADE.')
    args = parser.parse_args()
    print(args)

    auto_anomaly_ratio = args.auto_anomaly_ratio
    normalize = args.normalize
    test_only = args.test_only
    anomaly_ratio = args.anomaly_ratio
    batch_size = args.batch_size
    dataset = args.dataset
    epoch = args.epoch
    learning_rate = args.learning_rate
    train_set_ratio = args.train_set_ratio
    stride = args.stride
    window_size = args.window_size
    cuda = args.cuda
    gpu = args.gpu
    hidden_size = args.hidden_size
    n_blocks = args.n_blocks
    hidden_layers = args.hidden_layers
    batch_norm = args.batch_norm
    weight_decay = args.weight_decay
    save_dir = args.save_dir

    print('\033[0;34m program begin \033[0m')

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if not os.path.exists(save_dir):
        print('save directory not exists. created',save_dir)
        os.makedirs(save_dir)

    if auto_anomaly_ratio:
        if dataset == 'swat':
            anomaly_ratio = 0.1214
        elif dataset == 'wadi.old':
            anomaly_ratio = 0.0384
        elif dataset == 'wadi.new':
            anomaly_ratio = 0.0577
        else:
            anomaly_ratio = None
            print('invalid dataset')
            exit()
    else:
        anomaly_ratio = args.anomaly_ratio

    if platform.system() == 'Windows':
        data_dir = 'E:\\Pycharm Projects\\causal.dataset\\data'
        map_dir = 'E:\\Pycharm Projects\\causal.dataset\\maps\\npmap'
    else:
        data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
        map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'
    packfile = os.path.join(data_dir, dataset, 'raw.data.pkl')

    dataloader = Dataloader(pack_file=packfile, normalize=normalize, window_size=window_size, stride=stride,
                            test_set_only=test_only, train_set_ratio=train_set_ratio)

    train_set = dataloader.load_train_set()
    test_set = dataloader.load_test_set()
    labels = dataloader.load_labels_numpy()
    n_sensor = dataloader.load_n_sensors()
    print('train test labes shape',train_set.shape, test_set.shape, labels.shape)

    # print(train_set.shape, test_set.shape, labels.shape)
    # print(type(train_set), type(test_set), type(labels))

    ganf = Ganf(n_sensor, cuda, n_blocks, hidden_size, hidden_layers, batch_norm)
    ganf.tune_matrix_A(train_set, learning_rate, batch_size, weight_decay, cuda)
    ganf.train_model(train_set, learning_rate, batch_size, weight_decay, epoch, cuda)
    loss_list = ganf.evaluate(test_set, batch_size, cuda)
    print(np.where(np.isnan(loss_list) == True))
    print(np.min(loss_list), np.max(loss_list))
    print(type(labels),labels.shape)
    print('total anomalies',np.sum(labels))
    roc_val = roc_auc_score(labels, loss_list)
    roc_val_val = roc_auc_score(labels[:labels.shape[0] // 2], loss_list[:labels.shape[0] // 2])
    roc_val_test = roc_auc_score(labels[labels.shape[0] // 2:], loss_list[labels.shape[0] // 2:])
    recall,precision,f1=get_f1_scores(labels, loss_list, None)
    print('\033[0;31mroc:', roc_val)
    print('val roc, test roc', roc_val_val, roc_val_test)
    print('recall, precision, f1', recall, precision, f1, '\033[0m')

    f = open(os.path.join(save_dir, 'result.txt'), 'w')
    print('roc:', roc_val, file=f)
    print('val roc, test roc', roc_val_val, roc_val_test, file=f)
    print('recall, precision, f1', recall, precision, f1, file=f)
    print(args, file=f)
    f.close()
