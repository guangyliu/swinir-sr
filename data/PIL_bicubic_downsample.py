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
import mmcv
from mmsr.utils import FileClient
from mmsr.data.transforms import augment, mod_crop, totensor
import cv2
from data.util import paired_paths_from_ann_file

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in [2, 3]:
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

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
        # self.lr_dir = self.dataroot / 'LR_bicubic/X4'
        self.ref_dir = self.dataroot / 'ref'
        # self.ref_down_up_dir = self.dataroot / 'ref_down_up4'
        # self.lr_up_dir = self.dataroot / 'LR_up4'
        self.file_client = None
        self.io_backend_opt = dataset_opt['io_backend']
        if 'ann_file' in dataset_opt:
            self.paths = paired_paths_from_ann_file(
                [self.dataroot, self.dataroot], ['in', 'ref'],
                self.opt['ann_file'])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        scale = 4 #dataset_opt['scale']
        filename = self.filenames[index]

        # if self.opt['phase'] == 'train':
        hr_path = self.input_dir / f'{filename}.png'
        ref_path = self.ref_dir / f'{filename}.png'

        hr_bytes = self.file_client.get(hr_path, 'in')
        img_hr = mmcv.imfrombytes(hr_bytes).astype(np.float32) / 255.
        ref_bytes = self.file_client.get(ref_path, 'ref')
        img_ref = mmcv.imfrombytes(ref_bytes).astype(np.float32) / 255.

        gt_h, gt_w = 160, 160
        # some reference image in CUFED5_train have different sizes
        # resize reference image using PIL bicubic kernel
        img_ref = img_ref * 255
        img_ref = Image.fromarray(
            cv2.cvtColor(img_ref.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img_ref = img_ref.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref = cv2.cvtColor(np.array(img_ref), cv2.COLOR_RGB2BGR)
        img_ref = img_ref.astype(np.float32) / 255.
        # data augmentation
        img_hr, img_ref = augment([img_hr, img_ref], self.flip,
                                  self.rotation)

        # else:
        #     in_path = self.paths[index]['in_path']
        #     img_bytes = self.file_client.get(in_path, 'in')
        #     img_in = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        #     ref_path = self.paths[index]['ref_path']
        #     img_bytes = self.file_client.get(ref_path, 'ref')
        #     img_ref = mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.
        #
        #     # for testing phase, zero padding to image pairs for same size
        #     img_in = mod_crop(img_in, scale)
        #     img_in_gt = img_in.copy()
        #     img_ref = mod_crop(img_ref, scale)
        #     img_in_h, img_in_w, _ = img_in.shape
        #     img_ref_h, img_ref_w, _ = img_ref.shape
        #     padding = False
        #
        #     if img_in_h != img_ref_h or img_in_w != img_ref_w:
        #         padding = True
        #         target_h = max(img_in_h, img_ref_h)
        #         target_w = max(img_in_w, img_ref_w)
        #         img_in = mmcv.impad(
        #             img_in, shape=(target_h, target_w), pad_val=0)
        #         img_ref = mmcv.impad(
        #             img_ref, shape=(target_h, target_w), pad_val=0)
        #
        #     gt_h, gt_w, _ = img_in.shape

        # downsample image using PIL bicubic kernel
        lq_h, lq_w = gt_h // scale, gt_w // scale

        img_hr_pil = img_hr * 255
        img_hr_pil = Image.fromarray(
            cv2.cvtColor(img_hr_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_ref_pil = img_ref * 255
        img_ref_pil = Image.fromarray(
            cv2.cvtColor(img_ref_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))

        img_lr = img_hr_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_ref_down = img_ref_pil.resize((lq_w, lq_h), Image.BICUBIC)

        # bicubic upsample LR to original size
        img_lr_up = img_lr.resize((gt_w, gt_h), Image.BICUBIC)
        img_ref_down_up = img_ref_down.resize((gt_w, gt_h), Image.BICUBIC)

        img_lr = cv2.cvtColor(np.array(img_lr), cv2.COLOR_RGB2BGR)
        img_lr = img_lr.astype(np.float32) / 255.
        img_lr_up = cv2.cvtColor(np.array(img_lr_up), cv2.COLOR_RGB2BGR)
        img_lr_up = img_lr_up.astype(np.float32) / 255.
        img_ref_down = cv2.cvtColor(np.array(img_ref_down), cv2.COLOR_RGB2BGR)
        img_ref_down = img_ref_down.astype(np.float32) / 255.
        img_ref_down_up = cv2.cvtColor(np.array(img_ref_down_up), cv2.COLOR_RGB2BGR)
        img_ref_down_up = img_ref_down_up.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hr, img_lr, img_lr_up, img_ref, img_ref_down_up = totensor(  # noqa: E501
            [img_hr, img_lr, img_lr_up, img_ref, img_ref_down_up],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_hr': img_hr,
            'img_lr': img_lr,
            'img_ref': img_ref,
            'img_ref_down_up': img_ref_down_up,
            'img_lr_up': img_lr_up,
            'filename': filename
        }

        # if self.opt['phase'] != 'train':
        #     img_in_gt = totensor(img_in_gt, bgr2rgb=True, float32=True)
        #     return_dict['img_hr'] = img_in_gt
        #     return_dict['ref_path'] = ref_path
        #     return_dict['padding'] = padding
        #     return_dict['original_size'] = (img_in_h, img_in_w)

        return return_dict  # {'img_hr': img_hr, 'img_lr': img_lr, 'img_ref': img_ref, 'img_ref_down_up': img_ref_down_up, 'img_lr_up': img_lr_up, 'filename': filename}

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
