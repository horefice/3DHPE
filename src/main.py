import numpy as np
import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle

from nn import MyNN
from solver import Solver
from dataHandler import DataHandler
from utils.plotter import Plotter

## SETTINGS
parser = argparse.ArgumentParser(description='MyNET Implementation')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='F',
                    help='learning rate (default: 0.01)')
parser.add_argument('--val-size', type=float, default=0.2, metavar='F',
                    help='Validation set size ratio from training set (default: 0.2)')
parser.add_argument('--mnist', type=bool, default=False, metavar='B',
                    help='mnist for True, other dataset for False (default: False)')
parser.add_argument('--model', type=str, default='', metavar='S',
                    help='use previously saved model')
parser.add_argument('--plot', type=bool, default=False, metavar='B',
                    help='Plot train and validation histories (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

## LOAD DATA
if args.mnist:
  train_data = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
  test_data = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
else:
  train_data = DataHandler(path='../datasets/train.txt')
  test_data = DataHandler(path='../datasets/test.txt')

print("DATASET INFO.")
print("Train & validation size: %i" % len(train_data))
print("Test size: %i" % len(test_data))
print("Data dimensions:", train_data[0][0].size() if len(train_data) else "ERROR!","\n")

## LOAD MODELS & SOLVER
model = torch.load(args.model, map_location='cpu') if args.model else MyNN()
if args.cuda:
    model.cuda()
solver = Solver(optim_args={"lr": args.lr})

## TRAIN
if not args.model:
  train_sampler, val_sampler = DataHandler().subdivide_dataset(len(train_data), args.val_size, args.seed)

  train_loader = torch.utils.data.DataLoader(train_data,
                                            sampler=train_sampler,
                                            batch_size=args.batch_size,
                                            **kwargs)
  val_loader = torch.utils.data.DataLoader(train_data,
                                            sampler=val_sampler,
                                            batch_size=args.batch_size,
                                            **kwargs)
  solver.train(model, train_loader, val_loader, log_nth=args.log_interval, num_epochs=args.epochs)

## TEST
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.test_batch_size,
                                          shuffle=False, **kwargs)
test_acc,_ = solver.test(model, test_loader)

print('TESTING.')
print('Test accuracy: {:.2f}%\n'.format(test_acc*100))

## PLOT TRAINING
if args.plot:
  plotter = Plotter().plot_histories('(test_acc = ' + str(test_acc*100) + '%)')