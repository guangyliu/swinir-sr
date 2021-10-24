from pathlib import Path
import numpy as np
from data import common
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class CUFED5Dataset(Dataset):
    """
    Dataset class for CUFED5, which is a dataset provided the author of CIMR.
    """

    def __init__(self,
                 base_folder: Path):
        super(CUFED5Dataset, self).__init__()

        self.base_folder = Path(base_folder)
        self.hr_folder = Path(self.base_folder) / 'CUFED5'
        self.lr_folder = Path(base_folder) / 'CUFED5_LR'
        self.down_up_folder = Path(base_folder) / 'CUFED5_down_up_x4'
        self.filenames = sorted(list(set(
            [f.stem.split('_')[0] for f in self.hr_folder.glob('*.png')]
        )))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.warp = transforms.RandomAffine(
            degrees=(0, 0),
            translate=(0, 0.25),
            scale=(1.2, 2.0),
            resample=Image.BICUBIC
        )

    def __getitem__(self, index):
        def load_ref(resolution, f, i, warp=False):
            if resolution == 'hr':
                path = self.hr_folder
            elif resolution == 'lr':
                path = self.lr_folder
            elif resolution == 'down_up':
                path = self.down_up_folder
            # ref image
            img_ref = Image.open(path / f'{f}_{i}.png').convert('RGB')

            size = (x - (x % 4) for x in img_ref.size)
            img_ref = img_ref.resize(size, Image.BICUBIC)  # adjustment to x4

            if warp:
                img_ref = self.warp(img_ref)

            # down-upsampling ref image
            size = (x // 4 for x in img_ref.size)
            img_ref_blur = img_ref.resize(size, Image.BICUBIC) \
                .resize(img_ref.size, Image.BICUBIC)

            return {'ref': self.transforms(img_ref),
                    'ref_blur': self.transforms(img_ref_blur)}

        filename = self.filenames[index]


        img_hr = Image.open(self.hr_folder / f'{filename}_0.png').convert('RGB')
        img_lr = Image.open(self.lr_folder / f'{filename}.png').convert('RGB')
        img_down_up = Image.open(self.down_up_folder / f'{filename}.png').convert('RGB')

        img_hr = np.array(img_hr) #(332, 500, 3)
        img_down_up = np.array(img_down_up)
        img_lr = np.array(img_lr)

        window_size = 8
        # inference

        # pad input image to be a multiple of window_size
        h_old, w_old, _, = img_hr.shape
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_hr = torch.cat([img_hr, torch.flip(img_hr, [2])], 2)[:, :, :h_old + h_pad, :]
        img_hr = torch.cat([img_hr, torch.flip(img_hr, [3])], 3)[:, :, :, :w_old + w_pad]
        img_down_up = torch.cat([img_down_up, torch.flip(img_down_up, [2])], 2)[:, :, :h_old + h_pad, :]
        img_down_up = torch.cat([img_down_up, torch.flip(img_down_up, [3])], 3)[:, :, :, :w_old + w_pad]

        # Pad LR
        _, _, h_old_lr, w_old_lr = img_lr.size()
        h_pad = (h_old_lr // window_size + 1) * window_size - h_old_lr
        w_pad = (w_old_lr // window_size + 1) * window_size - w_old_lr
        img_lr = torch.cat([img_lr, torch.flip(img_lr, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lr = torch.cat([img_lr, torch.flip(img_lr, [3])], 3)[:, :, :, :w_old + w_pad]

        ref_dict = {i: load_ref(filename, i) for i in range(6)}



        img_lr, img_down_up, img_hr = common.set_channel([img_lr, img_down_up, img_hr], 3)
        lr_tensor, down_up_tensor, hr_tensor = common.np2Tensor([img_lr, img_down_up, img_hr], 1)

        return {'img_hr': hr_tensor,
                'img_down_up': down_up_tensor,
                'img_lr': lr_tensor,
                'ref': ref_dict,
                'filename': filename}

    def __len__(self):
        return len(self.filenames)


