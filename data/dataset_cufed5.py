from pathlib import Path
import numpy as np
from data import common
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUFED5Dataset(Dataset):
    """
    Dataset class for CUFED5, which is a dataset provided the author of CIMR.
    """

    def __init__(self,
                 dataroot: Path):
        super(CUFED5Dataset, self).__init__()

        self.dataroot = Path(dataroot)
        self.filenames = sorted(list(set(
            [f.stem.split('_')[0] for f in self.dataroot.glob('*.png')]
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
        def load_ref(f, i, warp=False):
            # ref image
            img_ref = Image.open(self.dataroot / f'{f}_{i}.png').convert('RGB')
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

        # HR image
        img_hr = Image.open(self.dataroot / f'{filename}_0.png').convert('RGB')
        size = (x - (x % 4) for x in img_hr.size)
        img_hr = img_hr.resize(size, Image.BICUBIC)  # adjustment to x4

        # LR image
        size = (x // 4 for x in img_hr.size)
        img_lr = img_hr.resize(size, Image.BICUBIC)

        # for feature searching
        img_in_up = img_lr.resize(img_hr.size, Image.BICUBIC)
        ref_dict = {0: load_ref(filename, 0)}
        # ref_dict = {i: load_ref(filename, i) for i in range(6)}
        # ref_dict.update({6: load_ref(filename, 0, warp=True)})
        # ref_dict = {6: load_ref(filename, 0, warp=True)}

        img_hr = np.array(img_hr)
        img_lr = np.array(img_lr)

        img_lr, img_hr = common.set_channel([img_lr, img_hr], 3)
        lr_tensor, hr_tensor = common.np2Tensor([img_lr, img_hr], 1)

        return {'img_hr': hr_tensor,
                'img_lr': lr_tensor,
                'img_in_up': self.transforms(img_in_up),
                'ref': ref_dict,
                'filename': filename}

    def __len__(self):
        return len(self.filenames)


