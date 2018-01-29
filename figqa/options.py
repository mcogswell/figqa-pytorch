import random
import os.path as pth
import argparse
import json
from time import gmtime, strftime

import torch.cuda

def parse_arguments():
    '''Return an argparse Namespace with command line params.'''
    parser = argparse.ArgumentParser(description='FigureQA Models in PyTorch')

    #####################################################################
    # Training
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--shuffle-train', type=int, default=1,
                        help='1 to present the train set in random order')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                        help='decay lr by this factor every epoch')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--env-name', default='figqa',
                        help='visdom environment name')
    parser.add_argument('--ngpus', default=-1, type=int,
                        help='number of gpus to use (default: all)')

    #####################################################################
    # Evaluation
    parser.add_argument('--result-dir', default='data/results/',
                        help='directory to save results in')
    parser.add_argument('--result-name', default='',
                        help='name to identify the context (esp model) for this result')

    #####################################################################
    # checkpoints
    parser.add_argument('--checkpoint-base', default='data/checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--start-from', help='load the model from this '
                        'checkpoint, overriding other model options')

    #####################################################################
    # Dataset
    parser.add_argument('--figqa-dir',
                        help='directory containing unzipped figureqa data')
    parser.add_argument('--figqa-pre',
                        help='directory containing preprocessed figqa output')
    parser.add_argument('--workers', type=int, default=20,
                        help='number of worker threads for data loader')
    parser.add_argument('--val-split',
                        help='split to evaluate', default='validation1',
                        choices=['no_annot_test1', 'no_annot_test2', 'validation1',
                                 'validation2', 'train1', 'sample_train1'])

    #####################################################################
    # Model
    parser.add_argument('--word-embed-dim', type=int, default=32)
    parser.add_argument('--ques-rnn-hidden-dim', type=int, default=256)
    parser.add_argument('--ques-num-layers', type=int, default=1)
    parser.add_argument('--rn-g-dim', type=int, default=256)
    parser.add_argument('--rn-f-dim', type=int, default=256)
    parser.add_argument('--img-net-dim', type=int, default=64)
    parser.add_argument('--model', default='rn',
                        choices=['lstm', 'cnn+lstm', 'rn'])

    args = parser.parse_args()
    # add extra arguments
    if args.ngpus == -1:
        args.cuda = torch.cuda.device_count()
    else:
        args.cuda = args.ngpus
    time_stamp = strftime('%d-%b-%y-%X-%a', gmtime())
    checkpoint_dir = pth.join(args.checkpoint_base, time_stamp)
    checkpoint_dir += '_{:0>6d}'.format(random.randint(0, 10e6))
    args.checkpoint_dir = checkpoint_dir
    return args

def model_args(args):
    '''Return a dict with only parameters relevant to the model.'''
    with open(pth.join(args.figqa_pre, 'vocab.json'), 'r') as f:
        vocab = json.load(f)
    return {
        'vocab_size': len(vocab['ind2word']),
        'word_embed_dim': args.word_embed_dim,
        'ques_rnn_hidden_dim': args.ques_rnn_hidden_dim,
        'ques_num_layers': args.ques_num_layers,
        'rn_g_dim': args.rn_g_dim,
        'rn_f_dim': args.rn_f_dim,
        'rn_bn': False,
        'model': args.model,
        'act_f': 'relu', # relu, elu
        'img_net_dim': args.img_net_dim,
    }
