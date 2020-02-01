import numpy as np
import tensorflow as tf
import argparse
import sys, os
sys.path.append("..")
from data.dataset import Dataset
from train import train
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_device', type=int, default=0,
                    help='choose which gpu to run')
parser.add_argument('--cross_data_rebuild', type=bool, default=False,
                    help='whether to rebuild cross data')
parser.add_argument('--data_rebuild', type=bool, default=False,
                    help='whether to rebuild train/test dataset')
parser.add_argument('--mat_rebuild', type=bool, default=False,
                    help='whether to rebuild` adjacent mat')
parser.add_argument('--processor_num', type=int, default=12,
                    help='number of processors when preprocessing data')
parser.add_argument('--batch_size', type=int, default=256,
                    help='size of mini-batch')
parser.add_argument('--train_neg_num', type=int, default=4,
                    help='number of negative samples per training positive sample')
parser.add_argument('--test_size', type=int, default=1,
                    help='size of sampled test data')
parser.add_argument('--test_neg_num', type=int, default=99,
                    help='number of negative samples for test')
parser.add_argument('--epochs', type=int, default=20,
                    help='the number of epochs')
parser.add_argument('--gnn_layers', nargs='?', default=[32,32,16,16,8],
                    help='the unit list of layers')
parser.add_argument('--mlp_layers', nargs='?', default=[32,16,8],
                    help='the unit list of layers')
parser.add_argument('--embedding_size', type=int, default=8,
                    help='the size for embedding user and item')
parser.add_argument('--topK', type=int, default=10,
                    help='topk for evaluation')
parser.add_argument('--regularizer_rate', type=float, default=0.01,
                    help='the regularizer rate')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout_message', type=float, default=0.1,
                    help='dropout rate of message')
parser.add_argument('--NCForMF', type=str, default='NCF',
                    help='method to propagate embeddings')

args = parser.parse_args()


if __name__ == '__main__':
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    dataset_s = Dataset('../data/cds_music/CDs_and_Vinyl.csv', args)
    dataset_t = Dataset('../data/cds_music/Digital_Music.csv', args)
    train(dataset_s, dataset_t, args)
