# *_*coding:utf-8 *_*
from pathlib import Path
import random
from data import common
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ReferenceDataset(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,
                 dataset_opt,
                 files: list,
                 rotation=False,
                 flip=False):

        super(ReferenceDataset, self).__init__()

        self.flip = flip
        self.rotation = rotation
        self.filenames = files
        self.dataroot = Path(dataset_opt['dataroot'])
        self.input_dir = self.dataroot / 'HR'
        self.lr_dir = self.dataroot / 'LR_bicubic/X4'
        self.ref_dir = self.dataroot / 'ref'
        self.ref_down_up_dir = self.dataroot / 'ref_down_up4'
        self.lr_up_dir = self.dataroot / 'LR_up4'  # to be generated

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_hr = Image.open(self.input_dir / f'{filename}.png').convert('RGB')
        img_lr = Image.open(self.lr_dir / f'{filename}.png').convert('RGB')
        img_ref = Image.open(self.ref_dir / f'{filename}.png').convert('RGB')
        img_ref_down_up = Image.open(self.ref_down_up_dir / f'{filename}.png').convert('RGB')
        img_lr_up = Image.open(self.lr_up_dir / f'{filename}.png').convert('RGB')

        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)
        img_ref = np.array(img_ref)
        img_ref_down_up = np.array(img_ref_down_up)
        img_lr_up = np.array(img_lr_up)
        # if img_hr.shape != (160, 160, 3):
        #     np.pad(img_hr, ((0, 160 - img_hr.shape[0]), (0, 160 - img_hr.shape[1]), (0, 0)), 'constant')
        #     np.pad(img_lr, ((0, 40 - img_lr.shape[0]), (0, 40 - img_lr.shape[1]), (0, 0)), 'constant')
        #     np.pad(img_ref, ((0, 160 - img_ref.shape[0]), (0, 160 - img_ref.shape[1]), (0, 0)), 'constant')
        #     np.pad(img_ref_down_up, ((0, 160 - img_ref_down_up.shape[0]), (0, 160 - img_ref_down_up.shape[1]), (0, 0)),
        #            'constant')
        #     np.pad(img_lr_up, ((0, 160 - img_lr_up.shape[0]), (0, 160 - img_lr_up.shape[1]), (0, 0)), 'constant')

        img_hr, img_lr, img_ref, img_ref_down_up, img_lr_up = common.set_channel(
            [img_hr, img_lr, img_ref, img_ref_down_up, img_lr_up], 3)
        img_hr, img_lr, img_ref, img_ref_down_up, img_lr_up = common.np2Tensor(
            [img_hr, img_lr, img_ref, img_ref_down_up, img_lr_up], 1)

        if self.rotation:
            # random rotate
            state = random.randint(0, 3)
            img_hr = img_hr.rot90(state, [1, 2])
            img_lr = img_lr.rot90(state, [1, 2])
            img_lr_up = img_lr_up.rot90(state, [1, 2])

        if self.flip:
            # random flip
            state = random.randint(0, 3)
            if state == 1:
                img_hr = img_hr.flip([1])
                img_lr = img_lr.flip([1])
                img_lr_up = img_lr_up.flip([1])

            elif state == 2:
                img_hr = img_hr.flip([2])
                img_lr = img_lr.flip([2])
                img_lr_up = img_lr_up.flip([2])

        return {'img_hr': img_hr, 'img_lr': img_lr, 'img_ref': img_ref, 'img_ref_down_up': img_ref_down_up,
                'img_lr_up': img_lr_up, 'filename': filename}

    def __len__(self):
        return len(self.filenames)


class ReferenceDatasetEval(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,
                 files: list,
                 dataroot: Path):
        super(ReferenceDatasetEval, self).__init__()

        self.filenames = files
        self.dataroot = Path(dataroot)
        self.input_dir = self.dataroot / 'HR'
        self.lr_dir = self.dataroot / 'LR_bicubic/X4'
        self.ref_dir = self.dataroot / 'ref'
        self.ref_down_up_dir = self.dataroot / 'ref_down_up4'
        self.lr_up_dir = self.dataroot / 'LR_up4'

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_hr = Image.open(self.input_dir / f'{filename}.png').convert('RGB')
        img_lr = Image.open(self.lr_dir / f'{filename}.png').convert('RGB')

        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)

        img_lr, img_hr = common.set_channel([img_lr, img_hr], 3)
        lr_tensor, hr_tensor = common.np2Tensor([img_lr, img_hr], 1)

        return {'img_hr': hr_tensor,
                'img_lr': lr_tensor,
                'maps': maps,
                'weights': weights,
                'filename': filename}

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    files = list([f.stem for f in Path('/data/amax/zwb/CIMR-SR/data/CUFED/').glob('map/*.npz')])

    dataset = ReferenceDataset(files, dataroot=Path('/data/amax/zwb/CIMR-SR/data/CUFED/'))
    dataloader = DataLoader(dataset, batch_size=4)

    for batch in dataloader:
        img_hr = batch['img_hr']  # torch.Size([4, 3, 160, 160])
        img_lr = batch['img_lr']  # torch.Size([4, 3, 40, 40])

        weights = batch['weights']  # torch.Size([4, 3, 38, 38])
        # correspondences = batch['correspondences']
        maps = batch['maps']
        print(img_hr.shape)
        print(img_lr.shape)
        print(weights.shape)
        # print(correspondences.shape)
        print(maps.keys())
        print('relu2_1: ', maps['relu2_1'].shape)  # torch.Size([4, 3, 128, 80, 80])
        print('-------------------------')
