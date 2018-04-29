import numpy as np
import torch
import scipy.io
# from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle

class DataHandler(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(DataHandler, self).__init__(*args, **kwargs)
        self.targets = scipy.io.loadmat('../../mpi_inf_3dhp/datasets/S1/Seq1/annot.mat')
            
        # subj_list = [f for f in self.root_dir_name.glob('*') if f.is_dir()]
        # for subj in subj_list:
        #     seq_list = [f for f in Path(subj).glob('*') if f.is_dir()]
        #     for seq in seq_list:
        #         for f in Path(seq).glob('test/*0_*.jpg'):
        #             self.image_names.append(f)
        #         # mat = scipy.io.loadmat('')
        #         # self.targets.append(mat['annot2d'][0])

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
        # img = Image.open(array[index]).convert('RGB')
        # center_crop = transforms.CenterCrop(240)
        # img = center_crop(img)
        img = to_tensor(super(DataHandler, self).__getitem__(index)[0])
        target = torch.from_numpy(self.targets['univ_annot3'][0][0][index]).float()

        return img, target

    def subdivide_dataset(self, val_size, seed):
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
