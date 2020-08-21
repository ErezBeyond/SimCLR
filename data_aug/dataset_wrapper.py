import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from data_aug import logged_transforms

np.random.seed(0)


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()

        train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
                                       transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_rockwell_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform_rockwell_withH()

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))
        train_dataset = datasets.ImageFolder('/home/erez/Downloads/rockwell/Images/for_training_456', transform=SimCLRDataTransformLogged(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform_rockwell_withH(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([logged_transforms.RandomResizedCrop(size=self.input_shape[0:2],
                                                                           scale=(0.8, 1.0), ratio=(5. / 6., 6. / 5.)),
                                              logged_transforms.RandomHorizontalFlip(),
                                              logged_transforms.RandomVerticalFlip(),
                                              logged_transforms.RandomApply([color_jitter], p=0.9),
                                              logged_transforms.RandomGrayscale(p=1.0),
                                              logged_transforms.GaussianBlur(kernel_size=11),
                                              logged_transforms.ToTensor()])
        return data_transforms

    def _get_simclr_pipeline_transform_rockwell(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0:2],
                                                                           scale=(0.8, 1.0), ratio=(5. / 6., 6. / 5.)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.9),
                                              transforms.RandomGrayscale(p=1.0),
                                              GaussianBlur(kernel_size=11),
                                              transforms.ToTensor()])
        return data_transforms

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.seed(0)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

class SimCLRDataTransformLogged(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(dict(img=sample, H=np.eye(3)))
        xj = self.transform(dict(img=sample, H=np.eye(3)))
        return xi, xj