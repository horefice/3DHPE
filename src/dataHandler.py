import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle

class DataHandler():
    def __init__(self, path=''):
        self.root_dir_name = os.path.dirname(path)
        self.image_names = []

        if os.path.isfile(path):
            with open(path) as f:
                self.image_names = f.read().splitlines()

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

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.bmp', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.bmp')).convert('RGB')
        # center_crop = transforms.CenterCrop(240)
        # img = center_crop(img)
        img = to_tensor(img)

        target = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_GT.bmp'))
        # target = center_crop(target)
        target = np.array(target, dtype=np.int64)

        target_labels = 1

        target_labels = torch.from_numpy(target_labels.copy())

        return img, target_labels

    def subdivide_dataset(self, size, val_size, seed):
        num_samples = int(size)
        indices = list(range(num_samples))
        split = int(np.floor(val_size * num_samples))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        return (train_sampler, val_sampler)