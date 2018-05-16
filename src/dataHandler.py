import numpy as np
import torch
import h5py
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

class DataHandler(datasets.ImageFolder):
  def __init__(self, *args, **kwargs):
    super(DataHandler, self).__init__(*args, **kwargs)
      
  def __getitem__(self, key):
    if isinstance(key, slice):
      # get the start, stop, and step from the slice
      return [self[ii] for ii in range(*key.indices(len(self)))]
    elif isinstance(key, int):
      # handle negative indices
      if key < 0:
        key += len(self)
      if key < 0 or key >= len(self):
        raise IndexError("The index (%d) is out of range." % key)
      # get the data from direct index
      return self.get_item_from_index(key)
    else:
      raise TypeError("Invalid argument type.")

  # def __len__(self):
  #     return len(self.image_names)

  def get_item_from_index(self, index):
    to_tensor = transforms.ToTensor()
    img, target = super(DataHandler, self).__getitem__(index)
    img = to_tensor(img)

    return img, target

class TrainDataHandler(DataHandler):
  def __init__(self, *args, **kwargs):
    super(TrainDataHandler, self).__init__(*args, **kwargs)
    self.targets = h5py.File('../datasets/annot.h5')

  def get_item_from_index(self, index):
    to_tensor = transforms.ToTensor()
    img = to_tensor(super(DataHandler, self).__getitem__(index)[0])
    # center_crop = transforms.CenterCrop(320)
    # img = center_crop(img)

    path = self.imgs[index][0]
    infos = ('.').join(path.split('/')[-1].split('.')[:-1]).split('_') #[SX,SeqX,Cam,frame]
    target = self.targets[infos[0]]['annot3_'+infos[1]+'_'+infos[2]][int(infos[3])-1]
    target = torch.from_numpy(target).float()

    return img, target

  def subdivide_dataset(self, val_size, shuffle=False, seed=1):
    num_samples = int(len(self))
    indices = list(range(num_samples))
    split = int(np.floor(val_size * num_samples))

    if shuffle:
      np.random.seed(seed)
      np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return (train_sampler, val_sampler)

class TestDataHandler(DataHandler):
  def __init__(self, *args, **kwargs):
    super(TestDataHandler, self).__init__(*args, **kwargs)
    self.targets = h5py.File('../datasets/annot.h5')

  def get_item_from_index(self, index):
    to_tensor = transforms.ToTensor()
    img = to_tensor(super(DataHandler, self).__getitem__(index)[0])
    # center_crop = transforms.CenterCrop(320)
    # img = center_crop(img)

    path = self.imgs[index][0]
    infos = ('.').join(path.split('/')[-1].split('.')[:-1]).split('_') #[SX,SeqX,Cam,frame]
    target = self.targets[infos[0]]['annot3_'+infos[1]+'_'+infos[2]][int(infos[3])-1]
    target = torch.from_numpy(target).float()

    return img, target
