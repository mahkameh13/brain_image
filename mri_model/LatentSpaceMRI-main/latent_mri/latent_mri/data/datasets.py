import glob
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self,
                 root,
                 mode='T1',
                 transforms=None):
        self.root = root
        self.mode = mode
        self.path_subjects = sorted(glob.glob(os.path.join(self.root, '*')))

        self.path_images = []
        for path_subject in self.path_subjects:
            subject_id = os.path.basename(path_subject)
            path_mode = os.path.join(path_subject,
                                     f'{subject_id}-{self.mode}.nii.gz')
            assert os.path.exists(path_mode), f'{path_mode} does not exist!'
            self.path_images.append(path_mode)
        self.num_subjects = len(self.path_images)
        self.indices = torch.arange(0, self.num_subjects)
        self.transforms = transforms

    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        path_image = self.path_images[idx]
        vol = nib.load(path_image)
        image = np.asanyarray(vol.dataobj)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transforms:
            image = self.transforms(image)

        label = self.indices[idx]

        return image, label
