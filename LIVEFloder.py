import torch.utils.data as data

from PIL import Image
import h5py
import os
import os.path
import scipy.io
import numpy as np
import random
import csv

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0,num):
        name = '%s%s%s' % ('img',str(index),'.bmp')
        filename.append(os.path.join(path,name))
        index = index + 1
    return filename
        


class LIVEFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.refpath = os.path.join(self.root, 'refimgs')
        self.refname = getFileName( self.refpath,'.bmp')

        self.imgname=[]
        self.labels = []
        self.csv_file = os.path.join(self.root, 'LIVEhist_new.txt')
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) 
                self.imgname.append(token[0])
                values = np.array(token[1:11], dtype='float32')
                values /= values.sum()
                self.labels.append(values)
        refnames_all = scipy.io.loadmat(os.path.join(self.root, 'refnames_all.mat'))
        self.refnames_all = refnames_all['refnames_all']

        self.orgs_files = scipy.io.loadmat(os.path.join(self.root, 'orgs.mat'))
        self.orgs =self.orgs_files['orgs']

        sample = []
        for i in range(0, len(index)):
            train_sel = (self.refname[index[i]] == self.refnames_all)
            train_sel = train_sel * ~self.orgs.astype(np.bool_)
            train_sel1 = np.where(train_sel == True)
            train_sel = train_sel1[1].tolist()
            for j, item in enumerate(train_sel):
                sample.append((os.path.join(self.root,'allimg', self.imgname[item]), self.labels[item]))


        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        length = len(self.samples)
        return length




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

if __name__ == '__main__':
    liveroot = '/home/gyx/DATA/imagehist/LIVE'
    index = list(range(0,29))
    random.shuffle(index)
    train_index = index[0:round(0.8*29)]
    test_index = index[round(0.8*29):29]
    trainset = LIVEFolder(root = liveroot, loader = default_loader, index = train_index)
    testset = LIVEFolder(root = liveroot, loader = default_loader, index = test_index)
