import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref

class ChairsModelNet(data.Dataset):
  def __init__(self, split, nViews):
    self.split = split
    self.nImages = 889 if self.split == 'train' else 100
    self.totalViews = 72 if self.split == 'train' else ref.totalViewsModelNet
    self.nViews = nViews
    self.base = 1 if self.split == 'train' else 890
    
    annot = np.zeros((self.nImages, self.totalViews, ref.J, 3))
    meta = np.zeros((self.nImages, self.totalViews, ref.metaDim))
    print 'Reading ModelNet annot and meta'
    for i in range(self.nImages):
      for v in range(self.totalViews):
        f_name = '{}/annots{}/{}/chair_{:04d}_{}.txt'.format(ref.ModelNet_dir, ref.ModelNet_version, split, self.base + i, v)
        tmp = np.loadtxt(f_name)
        annot[i][v] = tmp[:ref.J].copy()
        meta[i, v, 0] = 1 if split == 'train' else -1
        meta[i, v, 1] = i
        meta[i, v, 2] = v
        meta[i, v, 3:5] = tmp[ref.J, :2].copy() / 180. * np.arccos(-1)
    
    print '{} ModelNet on {} samples and {} views'.format(split, self.nImages, self.nViews)
    
    self.annot = annot.copy()
    self.meta = meta.copy()
  
  def LoadImage(self, index, view):
    path = '{}/images{}/{}/chair_{:04d}_{}.bmp'.format(ref.ModelNet_dir, ref.ModelNet_version, self.split, self.base + index, view)
    img = cv2.imread(path)
    return img

  def __getitem__(self, index):
    imgs = np.zeros((self.nViews, ref.imgSize, ref.imgSize, 3), dtype = np.float32)
    pts = np.zeros((self.nViews, ref.J, 3), dtype = np.float32)
    meta = np.zeros((self.nViews, ref.metaDim))
    for v in range(self.nViews):
      vv = np.random.randint(self.totalViews) if self.nViews < self.totalViews else v
      imgs[v] = self.LoadImage(index, vv).astype(np.float32).copy()
      pts[v] = self.annot[index, vv].copy()
      meta[v] = self.meta[index, vv].copy()

    imgs = imgs.transpose(0, 3, 1, 2) / 255.
    inp = torch.from_numpy(imgs)
    return inp, pts, meta
    
  def __len__(self):
    return self.nImages

