import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

"""MyNN"""
class MyNN(nn.Module):
  """
  A PyTorch implementation of a superclass network.
  """

  def __init__(self):
    """
    Initialize a new network.
    """
    super(MyNN, self).__init__()

  def forward(self, x):
    """
    Forward pass of the neural network. Should not be called manually but by
    calling a model instance directly.

    Inputs:
    - x: PyTorch input Variable
    """
    print("MyNN: Forward method should be overwritten!")
    return x

  def num_flat_features(self, x):
    """
    Computes the number of features if the spatial input x is transformed
    to a 1D flat input.
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

  @property
  def is_cuda(self):
    """
    Check if model parameters are allocated on the GPU.
    """
    return next(self.parameters()).is_cuda

  def save(self, path="../models/nn.pth"):
    """
    Save model with its parameters to the given path. Conventionally the
    path should end with "*.pth".

    Inputs:
    - path: path string
    """
    print('Saving model... %s' % path)
    torch.save(self.state_dict(), path)

"""MyNet"""
class MyNet(MyNN):
  """
  A PyTorch implementation of MyNet
  with the following architecture:

  resnet50 (except last layer) - fc1 - fc2
  """

  def __init__(self):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data.
    - num_filters: Number of filters to use in the convolutional layer.
    - filter_size: Size of filters to use in the convolutional layer.
    - hidden_dim: Number of units to use in the fully-connected hidden layer-
    - num_classes: Number of scores to produce from the final affine layer.
    - stride_conv: Stride for the convolution layer.
    - stride_pool: Stride for the max pooling layer.
    - weight_scale: Scale for the convolution weights initialization
    - pool: The size of the max pooling window.
    - dropout: Probability of an element to be zeroed.
    """
    super(MyNet, self).__init__()

    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
      param.requires_grad = False
    modules = list(resnet.children())[:-1] # delete the last fc layer.
    self.resnet = nn.Sequential(*modules)

    self.fc1 = nn.Linear(32768, 160)
    self.fc2 = nn.Linear(160, 84)

  def forward(self, x):
    out = self.resnet(x)
    out = Variable(out.data)
    out = out.view(out.size(0), -1)
    out = F.relu(self.fc1(out))
    return self.fc2(out)

class VNect(MyNN):
  def __init__(self):
    super(VNect, self).__init__()

    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
      param.requires_grad = False
    modules = list(resnet.children())[:-3] # until res4f
    self.resnet = nn.Sequential(*modules)

    self.res5a = Residual(1024, 1024)
    self.res5b_1 = nn.Conv2d(1024, 256, 1)
    self.res5b_2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
    self.res5b_3 = nn.Conv2d(128, 256, 1)

    self.res5c_1 = nn.ConvTranspose2d(256, 63, 4, stride=2, padding=1)
    self.res5c_2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)

    self.conv1 = nn.Conv2d(212, 128, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(128, 112, 1)

    # self.fc1 = nn.Linear(179200, 252)
    # self.fc2 = nn.Linear(252, 84)

    # self.init_weights_()

  # def init_weights_(self):
  #   import pickle
  #   model_weights = pickle.load(open('../../vnect.pkl', 'rb'), encoding='latin1')
    
  #   def init_weight(m,branch):
  #     m.weight.data.copy_(torch.from_numpy(model_weights[branch+'/weights']).permute(3,2,1,0))
  #     m.bias.data.copy_(torch.from_numpy(model_weights[branch+'/biases']))

  #   def init_weight_transp(m,branch):
  #     m.weight.data.copy_(torch.from_numpy(model_weights[branch+'/kernel']).permute(3,2,1,0))
    
  #   def init_weight_lin(m):
  #     torch.nn.init.xavier_uniform_(m.weight)
  #     m.bias.data.fill_(0.01)

  #   init_weight(self.res5a.conv1,'res5a_branch2a_new')
  #   init_weight(self.res5a.conv2,'res5a_branch2b_new')
  #   init_weight(self.res5a.conv3,'res5a_branch2c_new')
  #   init_weight(self.res5a.conv4,'res5a_branch1_new')
  #   init_weight(self.res5b_1,'res5b_branch2a_new')
  #   init_weight(self.res5b_2,'res5b_branch2b_new')
  #   init_weight(self.res5b_3,'res5b_branch2c_new')
  #   init_weight_transp(self.res5c_1,'res5c_branch1a')
  #   init_weight_transp(self.res5c_2,'res5c_branch2a')
  #   init_weight_lin(self.fc1)
  #   init_weight_lin(self.fc2)

  def forward(self, x):
    out = self.resnet(x) # (1024,w/16,h/16)

    out = self.res5a(out) # (1024,w/16,h/16)
    out = self.res5b_1(out)
    out = self.res5b_2(out)
    out = self.res5b_3(out) # (256,w/16,h/16)

    out1 = self.res5c_1(out) # (63,w/8,h/8)
    out2 = F.relu(self.res5c_2(out)) # (128,w/8,h/8)
    out = torch.cat((out1,out2),1)
    out3 = torch.add(torch.mul(out1[:,::3],out1[:,::3]),torch.mul(out1[:,1::3],out1[:,1::3]))
    out3 = torch.add(out3,torch.mul(out1[:,2::3],out1[:,2::3]))
    out3 = torch.sqrt(out3) # (28,w/8,h/8)
    out = torch.cat((out,out3),1) # (240,w/8,h/8)

    out = F.relu(self.conv1(out)) # (128,w/8,h/8)
    out = self.conv2(out) # (112,w/8,h/8)
    out = out.view(out.size(0),4,out.size(-1),out.size(-2),28)

    heatmap = out[:,0]
    map_x = out[:,1]
    map_y = out[:,2]
    map_z = out[:,3]

    hm_val,hm_idx = heatmap.max(1)
    _,hm_idx2 = hm_val.max(1)

    hm_idx1 = torch.zeros_like(hm_idx2)
    out_mod = torch.zeros(out.size(0),28*3)
    for j,x in enumerate(hm_idx2):
      for k,y in enumerate(x):
        hm_idx1[j,k] = hm_idx[j,y,k]
        out_mod[j,3*k] = map_x[j,hm_idx1[j,k],hm_idx2[j,k],k]
        out_mod[j,3*k+1] = map_y[j,hm_idx1[j,k],hm_idx2[j,k],k]
        out_mod[j,3*k+2] = map_z[j,hm_idx1[j,k],hm_idx2[j,k],k]

    return out_mod

class Residual(nn.Module):
  def __init__(self, numIn, numOut, resConv=False):
    super(Residual, self).__init__()

    self.numIn = numIn
    self.numOut = numOut
    self.resConv = resConv
    self.conv1 = nn.Conv2d(self.numIn, int(self.numOut/2), 1)
    self.bn1 = nn.BatchNorm2d(int(self.numOut/2))
    self.conv2 = nn.Conv2d(int(self.numOut/2), int(self.numOut/2), 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(int(self.numOut/2))
    self.conv3 = nn.Conv2d(int(self.numOut/2), self.numOut, 1)
    self.conv4 = nn.Conv2d(self.numIn, self.numOut, 1) 
    
  def forward(self, x):
    residual = x
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.conv3(out)

    if self.numIn != self.numOut | self.resConv:
      residual = self.conv4(x)
    
    return out + residual
