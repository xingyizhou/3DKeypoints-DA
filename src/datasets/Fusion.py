import torch.utils.data as data
import numpy as np
import torch
import cv2
import ref

class Fusion(data.Dataset):
  def __init__(self, SourceDataset, TargetDataset, nViews, targetRatio, totalTargetIm = 1):
    self.nViews = nViews
    self.targetRatio = targetRatio
    if ref.category == 'Chair':
      self.sourceDataset = SourceDataset('train', nViews)
      self.targetDataset = TargetDataset('train', nViews, totalTargetIm)
    else:
      self.sourceDataset = SourceDataset('train', nViews)
      self.targetDataset = TargetDataset('train', nViews)
    self.nSourceImages = len(self.sourceDataset)
    self.nTargetImages = int(self.nSourceImages * self.targetRatio)

    print '#Source images: {}, #Target images: {}'.format(self.nSourceImages, self.nTargetImages)
    
  def __getitem__(self, index):
    if index < self.nSourceImages: 
      return self.sourceDataset[index]
    else:
      return self.targetDataset[index - self.nSourceImages]

  def __len__(self):
    return (self.nSourceImages + self.nTargetImages)


