import torch
import ref
import numpy as np
import sys
from horn87 import horn87, RotMat

def accuracy(output, target, meta):
  batch_size = target.size(0)
  target = target.cpu().numpy()
  err, cnt = 0, 0
  for i in range(batch_size):
    if meta[i, 0] < 1 + ref.eps:
      cnt += 1
      for j in range(ref.J):
        err += ((output[i][j * 3] - target[i][j][0]) ** 2 + 
                (output[i][j * 3 + 1] - target[i][j][1]) ** 2 + 
                (output[i][j * 3 + 2] - target[i][j][2]) ** 2) ** 0.5
  if cnt > 0:
    return err / ref.J / cnt
  else:
    return 0
    
    
def accuracy_dis(output, target, meta):
  batch_size = target.size(0)
  target = target.cpu().numpy()
  output = output.cpu().numpy().reshape(batch_size, ref.J, 3)
  err, cnt = 0, 0
  for i in range(batch_size):
    if meta[i, 0] < 1 + ref.eps:
      cnt += 1
      R, t = horn87(output[i].transpose(), target[i].transpose())
      M = np.dot(R, output[i].transpose()).transpose()
      for j in range(ref.J):
        err += ((M[j, 0] - target[i][j][0]) ** 2 + 
                (M[j, 1] - target[i][j][1]) ** 2 + 
                (M[j, 2] - target[i][j][2]) ** 2) ** 0.5
  if cnt > 0:
    return err / ref.J / cnt
  else:
    return 0    
    
  
def shapeConsistency(points, meta, nViews, M_, split):
  if (M_ is None):
    return 0
  points = points.cpu().numpy().astype(np.float32)
  G = points.shape[0] / nViews
  points = points.reshape(G, nViews, ref.J, 3)
  meta = meta.numpy().reshape(G, nViews, ref.metaDim)
  p3 = np.zeros((G, nViews, 3, ref.J), dtype = np.float32)
  R = np.zeros((G, nViews, 3, 3), dtype = np.float32)
  loss = 0
  
  for g in range(G):
    if meta[g, 0, 0] < 1 + ref.eps:
      continue
    id = int(np.abs(meta[g, 0, 1]))
    M = M_[id].transpose(1, 0)
    for j in range(nViews):
      p2 = points[g, j].reshape(ref.J, 3).transpose(1, 0).copy()
      R[g, j], t = horn87(M, p2)
      p3[g, j] = (np.dot(t.reshape(3, 1), np.ones((1, ref.J))) + np.dot(R[g, j], M)).copy()
    loss += ((p2 - p3[g, j]) ** 2).sum() / ref.J / 3 / nViews
  return loss
  
  
