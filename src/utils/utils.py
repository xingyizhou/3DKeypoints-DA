import torch
import sys
import collections
import shutil
import numpy as np
import ref
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D


if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)

def collate_fn_cat(batch):
  "Puts each data field into a tensor with outer dimension batch size"
  if torch.is_tensor(batch[0]):
    out = None
    return torch.cat(batch, 0, out=out)
  elif type(batch[0]).__module__ == 'numpy':
    elem = batch[0]
    if type(elem).__name__ == 'ndarray':
      return torch.cat([torch.from_numpy(b) for b in batch], 0)
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  elif isinstance(batch[0], int):
    return torch.LongTensor(batch)
  elif isinstance(batch[0], float):
    return torch.DoubleTensor(batch)
  elif isinstance(batch[0], string_classes):
    return batch
  elif isinstance(batch[0], collections.Mapping):
    return {key: collate_fn_cat([d[key] for d in batch]) for key in batch[0]}
  elif isinstance(batch[0], collections.Sequence):
    transposed = zip(*batch)
    return [collate_fn_cat(samples) for samples in transposed]

  raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
                     
                     
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def show3D(ax, points, c = (255, 0, 0)):
    points = points.reshape(ref.J, 3)
    x, y, z = np.zeros((3, ref.J))
    for j in range(ref.J):
        x[j] = points[j, 0] 
        y[j] = - points[j, 1] 
        z[j] = - points[j, 2] 
    ax.scatter(z, x, y, c = c)
    for e in ref.edges:
        ax.plot(z[e], x[e], y[e], c =c)
        
        
