import torch
from torch.autograd import Function
import numpy as np
from utils.horn87 import horn87, RotMat
import ref

class ShapeConsistencyCriterion(Function):
  def __init__(self, nViews, supWeight = 1, unSupWeight = 1, M = None):
    super(ShapeConsistencyCriterion, self).__init__()
    self.nViews = nViews
    self.supWeight = supWeight
    self.unSupWeight = unSupWeight
    self.M = M

  def forward(self, input, target_, meta_):
    target = target_.numpy()
    G = target.shape[0] / self.nViews
    points = input.cpu().numpy().astype(np.float32)
    points = points.reshape(G, self.nViews, ref.J, 3)
    target = target.reshape(G, self.nViews, ref.J, 3)
    meta = meta_.numpy().reshape(G, self.nViews, ref.metaDim)
    output = 0
    self.p3 = np.zeros((G, self.nViews, 3, ref.J), dtype = np.float32)
    self.R = np.zeros((G, self.nViews, 3, 3), dtype = np.float32)

    for g in range(G):
      loss = 0
      if meta[g, 0, 0] > 1 + ref.eps:
        if (self.M is None):
          continue
        M = (self.M[int(meta[g, 0, 1])]).transpose(1, 0)
        for j in range(self.nViews):
          p2 = points[g, j].reshape(ref.J, 3).transpose(1, 0).copy()
          self.R[g, j], t = horn87(M, p2)
          self.p3[g, j] = (np.dot(t.reshape(3, 1), np.ones((1, ref.J))) + np.dot(self.R[g, j], M)).copy()
          loss += ((p2 - self.p3[g, j]) ** 2).sum() / ref.J / 3 / self.nViews
        output += self.unSupWeight * loss
      else:
        for v in range(self.nViews):
          loss += ((points[g, v] - target[g, v]) ** 2).sum() / ref.J / 3 / self.nViews
        output += self.supWeight * loss
    output = output / G
    self.save_for_backward(input, target_, meta_)
    return torch.ones(1) * output

  def backward(self, grad_output):
    input, target_, meta_ = self.saved_tensors
    grad_input = torch.zeros(input.shape)
    G = target_.shape[0] / self.nViews
    points = input.cpu().numpy()
    points = points.reshape(G, self.nViews, ref.J, 3)
    target = target_.numpy()
    target = target.reshape(G, self.nViews, ref.J, 3)
    meta = meta_.numpy().reshape(G, self.nViews, ref.metaDim)
    for g in range(G):
      if meta[g, 0, 0] > 1 + ref.eps:
        for j in range(self.nViews):
          p2 = points[g, j].copy()
          grad_input[g * self.nViews + j] +=  grad_output[0] * self.unSupWeight * 2 * torch.from_numpy(p2 - self.p3[g, j].transpose(1, 0)) / ref.J / 3 / self.nViews / G
      else:
        for v in range(self.nViews):
          grad_input[g * self.nViews + v] += grad_output[0] * self.supWeight * 2 * torch.from_numpy(points[g, v] - target[g, v]) / ref.J / 3 / self.nViews / G
    return grad_input, None, None

