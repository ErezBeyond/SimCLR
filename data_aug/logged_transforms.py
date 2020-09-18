from torchvision.transforms import functional as tvF
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from data_aug.gaussian_blur import GaussianBlur as GaussianBlurStd
from matplotlib import pyplot as plt
import cv2



class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img_dict):
        # fix parameter
        i, j, h, w = self.get_params(img_dict['img'], self.scale, self.ratio)
        T = np.array([[1.0, 0, -j], [0.0, 1.0, -i], [0.0, 0.0, 1]])
        S = np.array([[self.size[1]/w, 0, 0], [0.0, self.size[0]/h, 0], [0.0, 0.0, 1]])
        H = S @ T @ img_dict['H']
        img = tvF.resized_crop(img_dict['img'], i, j, h, w, self.size, self.interpolation)
        return dict(img=img, H=H)

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img_dict):
        if random.random() < self.p:
            img = tvF.hflip(img_dict['img'])
            HF = np.array([ [-1.0, 0, img.size[0]],
                   [0.0, 1.0, 0],
                   [0.0, 0.0, 1]])
            H = HF @ img_dict['H']
            img_dict = dict(img=img, H=H)
        return img_dict

class RandomVerticalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img_dict):
        if random.random() < self.p:
            img = tvF.vflip(img_dict['img'])
            VF = [[1.0, 0, 0],
                  [0.0, -1.0, img.size[1]],
                  [0.0, 0.0, 1]]
            H = VF @ img_dict['H']
            img_dict = dict(img=img, H=H)
        return img_dict

# below are some non-geometric transforms, so H is returned as is
class RandomApply(transforms.RandomApply):
    def __call__(self, img_dict):
        return dict(img=super(RandomApply, self).__call__(img_dict['img']), H=img_dict['H'])

class RandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, img_dict):
        return dict(img=super(RandomGrayscale, self).__call__(img_dict['img']), H=img_dict['H'])

class GaussianBlur(GaussianBlurStd):
    def __call__(self, img_dict):
        return dict(img=super(GaussianBlur, self).__call__(img_dict['img']), H=img_dict['H'])

class ToTensor(transforms.ToTensor):
    def __call__(self, img_dict):
        return dict(img=super(ToTensor, self).__call__(img_dict['img']), H=img_dict['H'])