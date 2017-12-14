import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref

class ChairsShapeNet(data.Dataset):
  def __init__(self, split, nViews, totalTargetIm = 1, fullTest = False):
    self.split = split
    assert not (split == 'train' and fullTest) 
    self.nImages = 830 if fullTest else (729 if split == 'train' else 100)
    self.base = 1 if fullTest else (1 if split == 'train' else 730)
    self.totalViews = ref.totalViewsShapeNet
    self.nViews = nViews
    self.viewBase = 0
    self.dataPath = ref.ShapeNet_dir
    self.perm = np.random.choice(self.nImages, size = self.nImages, replace = False)
    
    PI = np.arccos(-1)
    meta = np.zeros((self.nImages, self.totalViews, ref.metaDim))
    annots = np.zeros((self.nImages, self.totalViews, ref.J, 3))
    for i in range(self.nImages):
      for v in range(self.totalViews):
        file_name = '{}/annots{}/{}_{}.txt'.format(self.dataPath, ref.Annot_ShapeNet_version, self.base + i, v)
        tmp = np.loadtxt(file_name)
        annots[i][v] = tmp[:ref.J].copy()
        meta[i, v, 0] = 2 if split == 'train' else -2 #very tricky
        meta[i, v, 1] = i
        meta[i, v, 2] = v
        meta[i, v, 3:5] = tmp[ref.J, :2].copy() / 180.0 * PI
    self.meta = meta.copy()
    self.annots = annots.copy()
    print '{} Annotated-ShapeNet on {} samples and {} views'.format(split, self.nImages, self.nViews)
    
  def LoadImage(self, index, view):
    path = '{}/images_annot{}/{}_{}.bmp'.format(self.dataPath, ref.Annot_ShapeNet_version, index + self.base, view)
    img = cv2.imread(path)
    return img
  
  def shuffle(self):
    self.perm = np.random.choice(self.nImages, size = self.nImages, replace = False)
  
  def GetMetaInfo(self, index, view):
    return self.meta[index][view]
    
  def __getitem__(self, index):
    index = self.perm[index % self.nImages] if self.split == 'train' else index
    imgs = np.zeros((self.nViews, ref.imgSize, ref.imgSize, 3), dtype = np.float32)
    pts = np.zeros((self.nViews, ref.J, 3), dtype = np.float32)
    meta = np.zeros((self.nViews, ref.metaDim))
    for v in range(self.nViews):
      vv = np.random.randint(self.totalViews) if self.split == 'train' else v
      imgs[v] = self.LoadImage(index, vv).astype(np.float32).copy()
      pts[v] = self.annots[index, vv].copy()
      meta[v] = self.meta[index, vv].copy()
      
    imgs = imgs.transpose(0, 3, 1, 2) / 255.
    inp = torch.from_numpy(imgs)
    return inp, pts, meta
    
  def __len__(self):
    return self.nImages

