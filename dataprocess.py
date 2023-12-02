import os
import cv2
import glob
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class imgdata(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_info = self.get_img_info(img_dir)
        self.label_info = self.get_img_info(label_dir)

    def __getitem__(self, item):
        img = cv2.imread(self.img_info[item], -1)
        label = cv2.imread(self.label_info[item], -1)
        random = np.random.rand(1)

        img = self.augementation(img, random)
        label = self.augementation(label, random)

        return img, label

    def __len__(self):
        return len(self.img_info)

    def get_img_info(self, data_dir):
        data_info = sorted(glob.glob(os.path.join(data_dir, '*.*')))
        return data_info

    def augementation(self, data, random):
        flip_H = transforms.RandomHorizontalFlip(p=1)
        flip_V = transforms.RandomVerticalFlip(p=1)
        totensor = transforms.ToTensor()
        data = totensor(data)
        if random[0] < 0.3:
            img_flip = flip_V(data)
        elif random[0] > 0.7:
            img_flip = flip_H(data)
        else:
            img_flip = data

        return img_flip


class imgdata_val(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_info = self.get_img_info(img_dir)
        self.label_info = self.get_img_info(label_dir)

    def __getitem__(self, item):
        img = cv2.imread(self.img_info[item], -1)
        label = cv2.imread(self.label_info[item], -1)
        totensor = transforms.ToTensor()
        img = totensor(img)
        label = totensor(label)

        return img, label

    def __len__(self):
        return len(self.img_info)

    def get_img_info(self, data_dir):
        data_info = sorted(glob.glob(os.path.join(data_dir, '*.*')))

        return data_info


def image_crop(dir, target_size_h, target_size_w, stride1, stride2):

    """
    :param dir: The directory of the data to be cropped.
    :param target_size_h: Height of the cropped data.
    :param target_size_w: Width of the cropped data.
    :param stride: Stride of the cropping operation.
    :return:
    """

    dir_list = sorted(os.listdir(dir))
    for item in dir_list:
        dir_path = os.path.join(dir, item)
        img_list = sorted(glob.glob(os.path.join(dir_path, '*.*')))

        path_split = os.path.join(dir, item + '_split')

        # Create saving folder
        if not os.path.isdir(path_split):
            os.makedirs(path_split)

        # Crop images
        for k in img_list:
            image = cv2.imread(k, -1)
            image_shape = image.shape
            # print(image_shape)
            i = 0  # Counter
            for x in range(0, image_shape[0] - target_size_h + 1, stride1):
                for y in range(0, image_shape[1] - target_size_w + 1, stride2):
                    sub_img = image[x:x + target_size_h, y:y + target_size_w]
                    sub_img_path = os.path.join(path_split, k.split('/')[-1].split('.')[0] + '_' + str(i) + '.png')
                    cv2.imwrite(sub_img_path, sub_img)
                    i += 1

                    print(f'Cropping {item} data: {sub_img_path}')
    return True