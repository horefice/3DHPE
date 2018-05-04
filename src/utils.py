import numpy as np
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Plotter():
    """Loads and plots training history"""
    def __init__(self, path='../models/train_histories.npz'):
        self.path = path
        self.train_loss_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def load_histories(self):
        """
        Load training history with its parameters to self.path. Conventionally the
        path should end with "*.npz".
        """
        npzfile = np.load(self.path)
        self.train_loss_history = npzfile['train_loss_history']
        self.val_acc_history = npzfile['val_acc_history']
        self.val_loss_history = npzfile['val_loss_history']

    def plot_histories(self, extra_title=''):
        """
        Plot losses and accuracies from training and validation. Also plots a 
        smoothed curve for train_loss.

        Inputs:
        - extra_title: extra string to be appended to plot's title
        """
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle('Training histories' + extra_title)

        x_epochs = np.arange(1,len(self.val_loss_history)+1)*len(self.train_loss_history)/len(self.val_loss_history)

        cumsum = np.cumsum(np.insert(self.train_loss_history, 0, 0))
        N = 10 # Moving average size
        smoothed = (cumsum[N:] - cumsum[:-N]) / float(N)

        ax1.set_yscale('log')
        ax1.plot(self.train_loss_history, label="train")
        ax1.plot(x_epochs,self.val_loss_history, label="validation", marker='x')
        ax1.plot(smoothed, label="train_smoothed")
        ax1.legend()
        ax1.set_ylabel('log(loss)')
        ax1.set_xlabel('batch')
        
        ax2.plot(np.arange(1,len(self.val_acc_history)+1),self.val_acc_history, label="validation", marker='x')
        ax2.legend()
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')
        
        plt.show();
