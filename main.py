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

## SETTINGS
parser = argparse.ArgumentParser(description='MyNET Implementation')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--valid-size', type=float, default=0.2, metavar='VAL',
                    help='Validation set size ratio from training set (default: 0.2)')
parser.add_argument('--mnist', type=bool, default=False, metavar='D',
                    help='mnist for True, other dataset for False (default: False)')
parser.add_argument('--model', type=str, default='', metavar='M',
                    help='use previously saved model')
parser.add_argument('--plot', type=bool, default=False, metavar='M',
                    help='Plot train and validation histories (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

## LOAD DATA
train_data = DataHandler(path='datasets/train.txt')
test_data = DataHandler(path='datasets/test.txt')

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

num_train = len(train_data)
print("Train & validation size: %i" % (num_train))
print("Test size: %i" % len(test_data))
print("Data size: ", train_data[0][0].size(),"\n")

## LOAD MODELS & SOLVER
model = torch.load(args.model) if args.model else MyNN()
if args.cuda:
    model.cuda()
solver = Solver(optim_args={"lr": args.lr}, path='models/train_histories.npz')

## TRAIN
if args.model:
	solver.load_histories()
else:
	indices = list(range(num_train))
	split = int(np.floor(args.valid_size * num_train))

	if shuffle:
	    np.random.seed(args.seed)
	    np.random.shuffle(indices)

	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(train_data,
											  sampler=train_sampler,
	                                          batch_size=args.batch_size,
	                                          **kwargs)
	valid_loader = torch.utils.data.DataLoader(train_data,
											  sampler=valid_sampler,
	                                          batch_size=args.batch_size,
	                                          **kwargs)

	solver.train(model, train_loader, valid_loader, log_nth=args.log_interval, num_epochs=args.epochs)
	model.save(path='models/nn.model')

## TEST
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=args.test_batch_size,
                                          shuffle=False, **kwargs)

test_acc = solver.test(model, test_loader)

## PLOT TRAINING
if args.plot:
	solver.plot_histories(test_acc)