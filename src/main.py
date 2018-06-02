import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import torch
import datetime
import os

from nn import MyNet, VNect
from solver import Solver
from dataHandler import TrainDataHandler, TestDataHandler
import utils

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet Implementation')
parser.add_argument('-x', '--expID', type=str, default='', metavar='S',
                    help='Experiment ID')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, metavar='F',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--val-size', type=float, default=0.2, metavar='F',
                    help='val/(train+val) set size ratio (default: 0.2)')
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='use previously saved model')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from model')
parser.add_argument('--plot', action='store_true', default=False,
                    help='enables plot train and validation histories')
parser.add_argument('--visdom', action='store_true', default=False,
                    help='enables VISDOM')
parser.add_argument('--vnect', action='store_true', default=False,
                    help='uses VNect-like network')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging (default: 10)')

## SETUP
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.saveDir = os.path.join('../models/', args.expID)

if args.resume and not args.model:
  print("\n=> No model to resume training. Double-check arguments!")
  quit()

torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.benchmark = True
  kwargs = {'num_workers': 1, 'pin_memory': True}

if not os.path.exists(args.saveDir):
  os.makedirs(args.saveDir)

utils.writeArgsFile(args,args.saveDir)

## LOAD DATASETS
print('\nDATASET INFO.')

train_data = TrainDataHandler('../datasets/train/')
test_data = TestDataHandler('../datasets/test/')

print('Train & val. size: {} x {}'.format(len(train_data), train_data[0][0].size()))
print('Test size: {} x {}'.format(len(test_data), test_data[0][0].size()))

## LOAD MODEL & SOLVER
print('\nLOADING NETWORK & SOLVER.')

model = MyNet() if not args.vnect else VNect()
checkpoint = {}
if args.model:
  checkpoint.update(torch.load(args.model))
  model.load_state_dict(checkpoint['state_dict'])
if args.cuda:
  model.cuda()

solver = Solver(optim_args={'lr': args.lr}, saveDir=args.saveDir,
                vis=args.visdom)

print('LOADED.')

## TRAIN
if (not args.model) ^ args.resume:
  print('\nTRAINING.')

  train_sampler, val_sampler = train_data.subdivide_dataset(args.val_size,
                                                           shuffle=True,
                                                           seed=args.seed)

  train_loader = torch.utils.data.DataLoader(train_data,
                                            sampler=train_sampler,
                                            batch_size=args.batch_size,
                                            **kwargs)
  val_loader = torch.utils.data.DataLoader(train_data,
                                          sampler=val_sampler,
                                          batch_size=args.batch_size,
                                          **kwargs)
  solver.train(model, train_loader, val_loader, log_nth=args.log_interval,
              num_epochs=args.epochs, checkpoint=checkpoint)

  print('FINISH.')

## TEST
print('\nTESTING.')

test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=args.test_batch_size,
                                          shuffle=False, **kwargs)
test_acc, test_loss = solver.test(model, test_loader)

print('Test accuracy: {:.2%}'.format(test_acc))
print('Test loss: {:.2f}'.format(test_loss))
