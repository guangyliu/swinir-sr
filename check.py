import glob
from pathlib import Path
from PIL import Image
import numpy as np

# HR_path = '/data/amax/zwb/CIMR-SR/data/CUFED/ref_down_up4' #LR_up4
# files = list([f.stem for f in Path(HR_path).glob('*.png')])  # 11485 .png files
#
# for file in files:
#     file = file + '.png'
#     img_hr = Image.open(Path(HR_path) / file).convert('RGB')
#     img_hr = np.array(img_hr)
#     if img_hr.shape != (160, 160, 3):
#         print(file, img_hr.shape)

import os
path = '/data/amax/zwb/CIMR-SR/data/CUFED/ref_down_up4'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

n = 0
for i in fileList:
    # 设置旧文件名（就是路径+文件名）
    oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

    # 设置新文件名
    newname = path + os.sep + fileList[n].strip('ref')

    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1