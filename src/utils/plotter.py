import numpy as np
import matplotlib.pyplot as plt
import os.path

class Plotter():
    def __init__(self, path='../models/train_histories.npz'):
        self.path = path
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

        if os.path.isfile(self.path):
            self.load_histories()

    def load_histories(self):
        """
        Load training history with its parameters to self.path. Conventionally the
        path should end with "*.npz".
        """
        npzfile = np.load(self.path)
        self.train_loss_history = npzfile['train_loss_history']
        self.train_acc_history = npzfile['train_acc_history']
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

if __name__ == '__main__':
    plotter = Plotter()