import ref
import numpy as np
from utils.horn87 import horn87, RotMat, Dis
from progress.bar import Bar
from utils.debugger import Debugger
import torch

oo = 1e18

DEBUG = False

def getY(dataset):
  N = dataset.nImages
  Y = np.zeros((N, ref.J, 3))
  for i in range(N):
    y = dataset.annot[i, 0].copy()
    rotY, rotZ = dataset.meta[i, 0, 3:5].copy() / 180. * np.arccos(-1)
    Y[i] = np.dot(np.dot(RotMat('Z', rotZ), RotMat('Y', rotY)), y.transpose(1, 0)).transpose(1, 0)
  return Y
  
def initLatent(loader, model, Y, nViews, S, AVG = False):
  model.eval()
  nIters = len(loader)
  N = loader.dataset.nImages 
  M = np.zeros((N, ref.J, 3))
  bar = Bar('==>', max=nIters)
  sum_sigma2 = 0
  cnt_sigma2 = 1
  for i, (input, target, meta) in enumerate(loader):
    output = (model(torch.autograd.Variable(input)).data).cpu().numpy()
    G = output.shape[0] / nViews
    output = output.reshape(G, nViews, ref.J, 3)
    if AVG:
      for g in range(G):
        id = int(meta[g * nViews, 1])
        for j in range(nViews):
          RR, tt = horn87(output[g, j].transpose(), output[g, 0].transpose())
          MM = (np.dot(RR, output[g, j].transpose())).transpose().copy()
          M[id] += MM.copy() / nViews
    else:
      for g in range(G):
        #assert meta[g * nViews, 0] > 1 + ref.eps
        p = np.zeros(nViews)
        sigma2 = 0.1
        for j in range(nViews):
          for kk in range(Y.shape[0] / S):
            k = kk * S
            d = Dis(Y[k], output[g, j])
            sum_sigma2 += d 
            cnt_sigma2 += 1
            p[j] += np.exp(- d / 2 / sigma2)
            
        id = int(meta[g * nViews, 1])
        M[id] = output[g, p.argmax()]
        
        if DEBUG and g == 0:
          print 'M[id]', id, M[id], p.argmax()
          debugger = Debugger()
          for j in range(nViews):
            RR, tt = horn87(output[g, j].transpose(), output[g, p.argmax()].transpose())
            MM = (np.dot(RR, output[g, j].transpose())).transpose().copy()
            debugger.addPoint3D(MM, 'b')
            debugger.addImg(input[g * nViews + j].numpy().transpose(1, 2, 0), j)
          debugger.showAllImg()
          debugger.addPoint3D(M[id], 'r')
          debugger.show3D()
        
    
    Bar.suffix = 'Init    : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Dis: {dis:.6f}'.format(i, nIters, total=bar.elapsed_td, eta=bar.eta_td, dis = sum_sigma2 / cnt_sigma2)
    bar.next()
  bar.finish()
  #print 'mean sigma2', sum_sigma2 / cnt_sigma2
  return M
  
def stepLatent(loader, model, M_, Y, nViews, lamb, mu, S):
  model.eval()
  nIters = len(loader)
  if nIters == 0:
    return None
  N = loader.dataset.nImages
  M = np.zeros((N, ref.J, 3))
    
  bar = Bar('==>', max=nIters)
  ids = []
  Mij = np.zeros((N, ref.J, 3))
  err, num = 0, 0
  for i, (input, target, meta) in enumerate(loader):
    output = (model(torch.autograd.Variable(input)).data).cpu().numpy()
    G = output.shape[0] / nViews
    output = output.reshape(G, nViews, ref.J, 3)
    for g in range(G):
      #assert meta[g * nViews, 0] > 1 + ref.eps
      id = int(meta[g * nViews, 1])
      ids.append(id)
      #debugger = Debugger()
      for j in range(nViews):
        Rij, tt = horn87(output[g, j].transpose(), M_[id].transpose())
        Mj = (np.dot(Rij, output[g, j].transpose()).copy()).transpose().copy()
        err += ((Mj - M_[id]) ** 2).sum()
        num += 1
        Mij[id] = Mij[id] + Mj / nViews 
        #print 'id, j, nViews', id, j, nViews
        #debugger.addPoint3D(Mj, 'b')
      #debugger.addPoint3D(M_[id], 'r')
      #debugger.show3D()
      
    Bar.suffix = 'Step Mij: [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err : {err:.6f}'.format(i, nIters, total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  if mu < ref.eps:
    for id in ids:
      M[id] = Mij[id]
    return M
  
  Mi = np.zeros((N, ref.J, 3))
  bar = Bar('==>', max=len(ids))
  err, num = 0, 0
  for i, id in enumerate(ids):
    dis = np.ones((Y.shape[0])) * oo
    for kk in range(Y.shape[0] / S):
      k = kk * S
      dis[k] = Dis(Y[k], M_[id])
    minK = np.argmin(dis)
    Ri, tt = horn87(Y[minK].transpose(), M_[id].transpose())
    Mi_ = (np.dot(Ri, Y[minK].transpose())).transpose()
    Mi[id] = Mi[id] + Mi_
    err += dis[minK]
    num += 1
    Bar.suffix = 'Step Mi : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err: {err:.6f}'.format(i, len(ids), total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  
  tI = np.zeros((Y.shape[0] / S, 3))
  MI = np.zeros((N, ref.J, 3))
  cnt = np.zeros(N)
  bar = Bar('==>', max=Y.shape[0] / S)
  err, num = 0, 0
  for kk in range(Y.shape[0] / S):
    k = kk * S
    dis = np.ones((N)) * oo
    for id in ids:
      dis[id] = Dis(Y[k], M_[id])
    minI = np.argmin(dis)
    RI, tt = horn87(Y[k].transpose(1, 0), M_[minI].transpose(1, 0))
    MI_ = (np.dot(RI, Y[k].transpose())).transpose()
    err += ((MI_ - M_[minI]) ** 2).sum()
    num += 1
    MI[minI] = MI[minI] + MI_
    cnt[minI] += 1
    Bar.suffix = 'Step MI : [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:} | Err: {err:.6f}'.format(kk, Y.shape[0] / S, total=bar.elapsed_td, eta=bar.eta_td, err = err / num)
    bar.next()
  bar.finish()
  
  for id in ids:
    M[id] = (Mij[id] * (lamb / mu) + Mi[id] + MI[id] / (Y.shape[0] / S) * len(ids)) / (lamb / mu + 1 + cnt[id] / (Y.shape[0] / S) * (len(ids)))
  if DEBUG:
    for id in ids:
      debugger = Debugger()
      debugger.addPoint3D(M[id], 'b')
      debugger.addPoint3D(M_[id], 'r')
      debugger.show3D()
  return M
