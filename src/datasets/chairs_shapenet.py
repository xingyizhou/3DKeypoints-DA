import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref

class ChairsShapeNet(data.Dataset):
  def __init__(self, split, nViews):
    assert split == 'train', 'unsup shapenet error'
    self.split = split
    self.nImages = 2506
    self.base = 1
    self.totalViews = 36
    self.nViews = nViews
    self.viewBase = 0
    self.dataPath = ref.ShapeNet_dir
    self.perm = np.random.choice(self.nImages, size = self.nImages, replace = False)
    
    PI = np.arccos(-1)
    meta = np.zeros((self.nImages, self.totalViews, ref.metaDim))
    print 'Reading meta'
    for i in range(self.nImages):
      for v in range(self.totalViews):
        file_name = '{}/meta{}/{}_{}.txt'.format(self.dataPath, ref.ShapeNet_version, self.base + i, v)
        tmp = np.loadtxt(file_name)
        meta[i, v, 0] = 2 if split == 'train' else -2
        meta[i, v, 1] = i
        meta[i, v, 2] = v
        meta[i, v, 3:5] = tmp[0, :2].copy() / 180.0 * PI
    self.meta = meta.copy()
    print '{} Unsup-ShapeNet on {} samples and {} views'.format(split, self.nImages, self.nViews)
    
  def LoadImage(self, index, view):
    path = '{}/images_unsup{}/{}_{}.bmp'.format(self.dataPath, ref.ShapeNet_version, index + self.base, view)
    img = cv2.imread(path)
    return img  
    
  def shuffle(self):
    self.perm = np.random.choice(self.nImages, size = self.nImages, replace = False)

  def __getitem__(self, index):
    index = self.perm[index]
    imgs = np.zeros((self.nViews, ref.imgSize, ref.imgSize, 3), dtype = np.float32)
    pts = np.zeros((self.nViews, ref.J, 3), dtype = np.float32)
    meta = np.zeros((self.nViews, ref.metaDim))
    for v in range(self.nViews):
      vv = np.random.randint(self.totalViews)
      imgs[v] = self.LoadImage(index, vv).astype(np.float32).copy()
      meta[v] = self.meta[index, vv].copy()
    imgs = imgs.transpose(0, 3, 1, 2) / 255.
    inp = torch.from_numpy(imgs)
    return inp, pts, meta
    
  def __len__(self):
    return self.nImages

