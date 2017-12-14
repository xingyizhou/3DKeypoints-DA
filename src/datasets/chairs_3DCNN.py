import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref
import os

class Chairs3DCNN(data.Dataset):
  def __init__(self, split, nViews, _ = 1):
    self.split = split
    self.viewBase = 1
    self.base = 1 if split == 'train' else 10
    self.nImages = 9 if split == 'train' else 3
    self.nViews = nViews
    self.viewBase = 1
    self.dataPath = ref.DCNN_dir
    self.version = ref.DCNN_version
    self.maxViews = 165
    self.totalViews = np.zeros((self.nImages), np.int32)
    
    
    annot = np.zeros((self.nImages, self.maxViews, ref.J, 3))
    meta = np.zeros((self.nImages, self.maxViews, ref.metaDim))
    for i in range(self.nImages):
      cnt = 0
      for v in range(self.maxViews):
        f_name = '{}/annots{}/{}_{}.txt'.format(self.dataPath, self.version, self.base + i, self.viewBase + v)
        if not os.path.exists(f_name):
          break
        cnt += 1
        tmp = np.loadtxt(f_name)
        annot[i][v] = tmp[:ref.J].copy()
        meta[i, v, 0] = 3 if split == 'train' else -3
        meta[i, v, 1] = i
        meta[i, v, 2] = v
        meta[i, v, 3:5] = tmp[ref.J, :2].copy() / 180. * np.arccos(-1)
      self.totalViews[i] = cnt
    self.annot = annot.copy()
    self.meta = meta.copy()
    print self.totalViews
    print '{} RealData on {} samples and {} views'.format(split, self.nImages, self.nViews)
    
  def LoadImage(self, index, view):
    path = '{}/images{}/{}_{}.bmp'.format(self.dataPath, self.version, index + self.base, view + self.viewBase)
    img = cv2.imread(path)
    return img
  
  def shuffle(self):
    pass
  
  def __getitem__(self, index):
    index = np.random.randint(self.nImages) if self.split == 'train' else index % self.nImages
    imgs = np.zeros((self.nViews, ref.imgSize, ref.imgSize, 3), dtype = np.float32)
    pts = np.zeros((self.nViews, ref.J, 3), dtype = np.float32)
    meta = np.zeros((self.nViews, ref.metaDim))
    for v in range(self.nViews):
      vv = np.random.randint(self.totalViews[index]) if self.split == 'train' else self.totalViews[index] * v / self.nViews
      imgs[v] = self.LoadImage(index, vv).astype(np.float32).copy()
      pts[v] = self.annot[index, vv].copy()
      meta[v] = self.meta[index, vv].copy()
    imgs = imgs.transpose(0, 3, 1, 2) / 255.
    inp = torch.from_numpy(imgs)
    return inp, pts, meta
    
  def __len__(self):
    return self.nImages

