import numpy as np
from random import shuffle
import time

import torch
from torch.autograd import Variable
from viz import Viz

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), vis=False):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.visdom = Viz() if vis else False

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(filter(lambda p: p.requires_grad,model.parameters()), **self.optim_args)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        self._reset_histories()
        self.best_model = None
        iter_per_epoch = len(train_loader)
        best_val_acc = 0.0

        if self.visdom:
            iter_plot = self.visdom.create_plot('Epoch', 'Loss', 'Train Loss', {'ytype':'log'})

        print('\nSTART TRAINING.')
        ########################################################################
        # The log should like something like:                                  #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN   loss: 0.560/1.374                              #
        #   [Epoch 1/5] VAL acc/loss: 0.539/1.310                              #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            train_loss = 0
            last_time = time.time()

            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.4f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))

                    if self.visdom:
                        self.visdom.update_plot(
                            x=epoch + i / iter_per_epoch,
                            y=train_loss,
                            window=iter_plot,
                            type_upd="append")

            if log_nth:
                print('[Epoch %d/%d] TRAIN   loss: %.4f (%d ms)' % (epoch + 1,
                                                           num_epochs,
                                                           train_loss,
                                                           (time.time()-last_time)*1000.0))

            # VALIDATION
            if len(val_loader):
                val_acc, val_loss = self.test(model, val_loader)
                self.val_acc_history.append(val_acc)
                self.val_loss_history.append(val_loss)
                if log_nth:
                    print('[Epoch %d/%d] VAL acc/loss: %.4f/%.4f' % (epoch + 1,
                                                                       num_epochs,
                                                                       val_acc,
                                                                       val_loss))

                # Reduce LR progressively
                scheduler.step(val_loss)

                # Update best model to the one with highest validation set accuracy
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    self.best_model = model
            else:
                self.best_model = model

        self.best_model.save(path='../models/nn.pth')
        self._save_histories(path='../models/train_histories.npz')
        print('FINISH.')

    def test(self, model, test_loader):
        """
        Test a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - test_loader: test data in torch.utils.data.DataLoader
        """
        test_losses = []
        test_scores = 0
        model.eval()

        for inputs, targets in test_loader:
            inputs, targets = Variable(inputs), Variable(targets)
            if model.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model.forward(inputs)
            loss = self.loss_func(outputs, targets)
            test_losses.append(loss.data.cpu().numpy())

            # diff = (outputs - targets).pow(2).mean()
            # if diff.data.cpu().numpy() < 35000:
            #     test_scores += 1
            
        test_acc, test_loss = test_scores / len(test_loader), np.mean(test_losses)
        return test_acc, test_loss

    def _save_histories(self, path="save.npz"):
        """
        Save training history with its parameters to self.path. Conventionally the
        path should end with "*.npz".
        """
        print('Saving training histories... %s' % path)
        np.savez(path, train_loss_history=self.train_loss_history,
                        val_loss_history=self.val_loss_history,
                        val_acc_history=self.val_acc_history)
