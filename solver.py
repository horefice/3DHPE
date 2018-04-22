import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), path='models/train_histories.npz'):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.path = path

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
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
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        self.best_model = None
        iter_per_epoch = len(train_loader)
        best_val_acc = 0.0

        print('START TRAIN.')
        ########################################################################
        # The log should like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING
            model.train()

            for i, (inputs, targets) in enumerate(train_loader, 1):
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                loss = self.loss_func(outputs.view(targets.size(0), -1), targets.view(-1))
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

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0
            targets_mask = targets >= 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.4f/%.4f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
            
            # VALIDATION
            if len(val_loader):
                val_losses = []
                val_scores = []
                model.eval()
                for inputs, targets in val_loader:
                    inputs, targets = Variable(inputs), Variable(targets)
                    if model.is_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = model.forward(inputs)
                    loss = self.loss_func(outputs, targets)
                    val_losses.append(loss.data.cpu().numpy())

                    _, preds = torch.max(outputs, 1)

                    # Only allow images/pixels with target >= 0 e.g. for segmentation
                    targets_mask = targets >= 0
                    scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                    val_scores.append(scores)

                val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
                self.val_acc_history.append(val_acc)
                self.val_loss_history.append(val_loss)
                if log_nth:
                    print('[Epoch %d/%d] VAL   acc/loss: %.4f/%.4f' % (epoch + 1,
                                                                       num_epochs,
                                                                       val_acc,
                                                                       val_loss))

                # Update best model to the one with highest validation set accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_model = model

        print('FINISH.\n')
        self.best_model.save(path='models/nn.model')
        self._save_histories()

    def test(self, model, test_loader):
        """
        Test a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - test_loader: test data in torch.utils.data.DataLoader
        """
        test_scores = []
        model.eval()
        for inputs, targets in test_loader:
            inputs, targets = Variable(inputs), Variable(targets)
            if model.is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            targets_mask = targets >= 0
            test_scores.append(np.mean((preds == targets)[targets_mask].data.cpu().numpy()))
            
        test_acc = np.mean(test_scores)*100
        print('Test set accuracy: {:.2f}%'.format(test_acc))

        return test_acc

    def _save_histories(self):
        """
        Save training history with its parameters to self.path. Conventionally the
        path should end with "*.npz".
        """
        print('Saving training histories... %s' % self.path)
        np.savez(self.path, train_loss_history=self.train_loss_history,
                        train_acc_history=self.train_acc_history,
                        val_loss_history=self.val_loss_history,
                        val_acc_history=self.val_acc_history)

    def load_histories(self):
        """
        Load training history with its parameters to self.path. Conventionally the
        path should end with "*.npz".
        """
        self._reset_histories()

        npzfile = np.load(self.path)
        self.train_loss_history = npzfile['train_loss_history']
        self.train_acc_history = npzfile['train_acc_history']
        self.val_acc_history = npzfile['val_acc_history']
        self.val_loss_history = npzfile['val_loss_history']

    def plot_histories(self, test_acc='xx.xx'):
        """
        Plot losses and accuracies from training and validation. Also plots a 
        smoothed curve for train_loss.

        Inputs:
        - test_acc: test set accuracy
        """
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle('Training histories (test_acc = ' + str(test_acc) + '%)')

        x_epochs = np.arange(1,len(self.val_loss_history)+1)*len(self.train_loss_history)/len(self.val_loss_history)

        cumsum = np.cumsum(np.insert(self.train_loss_history, 0, 0))
        N = 100 # Moving average size
        smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)

        ax1.set_yscale('log')
        ax1.plot(self.train_loss_history, label="train")
        ax1.plot(x_epochs,self.val_loss_history, label="validation", marker='x')
        ax1.plot(smoothed, label="train_smoothed")
        ax1.legend()
        ax1.set_ylabel('log(loss)')
        ax1.set_xlabel('batch')
        
        ax2.plot(np.arange(1,len(self.train_acc_history)+1),self.train_acc_history, label="train", marker='d')
        ax2.plot(np.arange(1,len(self.val_acc_history)+1),self.val_acc_history, label="validation", marker='x')
        ax2.legend()
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')
        
        plt.show();
