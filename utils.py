import torch
import math
from torchvision import transforms
import glob
import os
import cv2
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cal_psnr(img1, img2):
    img1 = img1*torch.mean(img2)/torch.mean(img1)
    mse = torch.mean((img1 - img2 ) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 ** 2 / mse)


def cal_s(img):
    img0 = img[:, 0, :, :]
    img45 = img[:, 1, :, :]
    img90 = img[:, 2, :, :]
    img135 = img[:, 3, :, :]
    S0 = (img0 + img45 + img90 + img135) * 0.5

    S1 = img0 - img90

    S2 = img45 - img135

    return S0, S1, S2


def cal_dolp(S0, S1, S2):
    DoLP = torch.sqrt((S1 ** 2 + S2 ** 2) / ((S0 + 0.000001) ** 2))
    # DoLP = torch.clip(DoLP*255 , 0, 255)

    return DoLP


def cal_aop(S1, S2):
    AoP = 1 / 2 * torch.atan2(S2, S1)
    AoP = (AoP + math.pi / 2.0) / (math.pi)
    # AoP = AoP * (255.0 / torch.nanmax(AoP))

    return AoP


def todofp(img):
    img0 = img[:, 0, :, :]
    img45 = img[:, 1, :, :]
    img90 = img[:, 2, :, :]
    img135 = img[:, 3, :, :]
    B = torch.zeros(img.shape[0], 1, img.shape[2] * 2, img.shape[3] * 2)
    B[:, :, 1:B.shape[2]:2, 1:B.shape[3]:2] = img0
    B[:, :, 0:B.shape[2]:2, 1:B.shape[3]:2] = img45
    B[:, :, 0:B.shape[2]:2, 0:B.shape[3]:2] = img90
    B[:, :, 1:B.shape[2]:2, 0:B.shape[3]:2] = img135

    return B


if __name__ == '__main__':
    totensor = transforms.ToTensor()
    file = './input'
    list = sorted(glob.glob(os.path.join(file, '*.png')))
    for k in list:
        img = totensor((cv2.imread(k, -1)))
        img.unsqueeze_(0)
        # label = totensor(Image.open('E:\\label_merge\\21.png'))
        # label.unsqueeze_(0)
        # print(psnr1(img,label).item())
        # dofp
        dofp = todofp(img)
        # print(dofp.shape)
        dofp.squeeze_(0)
        dofp.squeeze_(0)
        dofp = dofp.numpy() * 255
        print(dofp)
        cv2.imwrite('./input_dofp/' + k.split('/')[-1].split('.')[0] + '_dofp.png', dofp)
    #
    # S0,S1,S2=cal_s(img)
    #
    # #dolp
    # DOLP=cal_dolp(S0,S1,S2)
    # DOLP.squeeze_(0)
    # dolp=toPIL(DOLP)
    # dolp.save('E:\\21_dolp.png')
    # #aop
    # aop=cal_aop(S1,S2)
    # aop.squeeze_(0)
    # aop=toPIL(aop)
    # aop.save('E:\\21_aop.png')
