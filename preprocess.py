import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from render import show_images


random_seed = 42
torch.manual_seed(random_seed);


def show_batch(batch, titles=[], imsize=(10,10), cmap=None, ncols=2,
                keep_ticks=False, font_size=16):
    # batch_numpy = batch.permute(0,2,3,1).numpy()
    ls_imgs = list(batch)
    for i, im in enumerate(ls_imgs):
        # im = 255*((im-im.min())/(im.max()-im.min()))
        # ls_imgs[i] = im.astype(np.uint8)
        #print('ls_imgs[i,:,:,:].shape: ', ls_imgs[i].shape)
        ls_imgs[i] = np.array(tt.ToPILImage()(ls_imgs[i]))

    show_images(ls_imgs, titles, imsize, ncols, cmap, font_size,
                keep_ticks)


def create_dataloader(dataset_path, batch_size=64, shuffle=False):
    transform = tt.Compose([
        tt.ToTensor()
    ])
    dataset = ImageFolder(dataset_path, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dl





if __name__=="__main__":
    images_path = "dataA/dataA"
    images_paths = glob(images_path+"/CameraRGB/*")
    images = []
    titles = []
    for i, img_path in enumerate(images_paths[:50]):
        images.append(plt.imread(img_path))
        titles.append("title_"+str(i))

    #show_images(images, titles, ncols=10)

    train_dl = create_dataloader(images_path)
    #print('train_dl: ', train_dl)

    for img, seg in train_dl:
        #print('img.shape: ', img.shape)
        #print('seg.shape: ', seg.shape)
        show_batch(img, ncols = 16)
        #show_batch(seg)
        #print('seg: ', seg)
        break


