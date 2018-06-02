import sys
import numpy as np
import torch
from torch.autograd import Variable
import os

sys.path.append("../src/")
from nn import MyNet, VNect
from solver import Solver
from dataHandler import TrainDataHandler, TestDataHandler
import utils

class TestClass(object):
  batch_size = 2
  net_input = Variable(torch.from_numpy(np.random.rand(batch_size,3,320,320)).float())

  def forward(self, net):
    net.eval()
    out = net.forward(self.net_input)
    return out

  def test_mynet(self):
    net = MyNet()
    out = self.forward(net)
    assert out.shape == torch.Size([self.batch_size,84])

  def test_vnect(self):
    net = VNect()
    out = self.forward(net)
    assert out.shape == torch.Size([self.batch_size,84])
