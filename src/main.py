# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import torch

from nn import MyNet
from solver import Solver
from dataHandler import TrainDataHandler, TestDataHandler
from utils import Plotter

## SETTINGS
parser = argparse.ArgumentParser(description='MyNet Implementation')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, metavar='F',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--val-size', type=float, default=0.2, metavar='F',
                    help='validation set size ratio from training set (default: 0.2)')
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='use previously saved model')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from model')
parser.add_argument('--plot', action='store_true', default=False,
                    help='enables plot train and validation histories')
parser.add_argument('--visdom', action='store_true', default=False,
                    help='enables VISDOM')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status (default: 10)')

## PREPARE ARGS
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.resume and not args.model:
  print("\n=> No model to resume training. Double-check arguments!")
  quit()

torch.manual_seed(args.seed)
kwargs = {}
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)
  kwargs = {'num_workers': 1, 'pin_memory': True}

## LOAD DATA
train_data = TrainDataHandler('../datasets/train/')
test_data = TestDataHandler('../datasets/test/')

print("\nDATASET INFO.")
print("Train & validation size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Data dimensions:", train_data[0][0].size())

## LOAD MODELS & SOLVER
model = MyNet()
checkpoint = {}
if args.model:
  checkpoint = torch.load(args.model)
  model.load_state_dict(checkpoint['state_dict'])
if args.cuda:
  model.cuda()
solver = Solver(optim_args={"lr": args.lr},
                loss_func=torch.nn.MSELoss(), vis=args.visdom)

## TRAIN
if (not args.model) ^ args.resume:
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

## TEST
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.test_batch_size,
                                          shuffle=False, **kwargs)
test_acc, test_loss = solver.test(model, test_loader)

print('\nTESTING.')
print('Test accuracy: {:.2f}%'.format(test_acc*100))
print('Test loss: {:.2f}'.format(test_loss))

## PLOT TRAINING
if args.plot:
  plotter = Plotter()
  plotter.load_histories()
  plotter.plot_histories('(test_acc = {:.2f}%)'.format(test_acc*100))
