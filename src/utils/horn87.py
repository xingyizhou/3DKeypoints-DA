import numpy as np
from scipy.stats import ortho_group

def horn87(pointsS, pointsT):
  centerS = np.zeros((3))#pointsS.mean(axis = 1)
  centerT = np.zeros((3))#pointsT.mean(axis = 1)
  for i in range(pointsS.shape[1]):
    pointsS[:, i] = pointsS[:, i] - centerS
    pointsT[:, i] = pointsT[:, i] - centerT
  M = np.dot(pointsS, pointsT.transpose(1, 0))
  N = np.array([[M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]], 
                [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[0, 2] + M[2, 0]], 
                [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], M[1, 1] - M[0, 0] - M[2, 2], M[1, 2] + M[2, 1]], 
                [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], M[2, 2] - M[0, 0] - M[1, 1]]])
  v, u = np.linalg.eig(N)
  id = v.argmax()

  q = u[:, id]
  R = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
  t = centerT - np.dot(R, centerS)

  return R.astype(np.float32), t.astype(np.float32) 

def RotMat(axis, ang):
  s = np.sin(ang)
  c = np.cos(ang)
  res = np.zeros((3, 3))
  if axis == 'Z':
    res[0, 0] = c
    res[0, 1] = -s
    res[1, 0] = s
    res[1, 1] = c
    res[2, 2] = 1
  elif axis == 'Y':
    res[0, 0] = c
    res[0, 2] = s
    res[1, 1] = 1
    res[2, 0] = -s
    res[2, 2] = c
  elif axis == 'X':
    res[0, 0] = 1
    res[1, 1] = c
    res[1, 2] = -s
    res[2, 1] = s
    res[2, 2] = c
  return res

def Dis(X, Y):
  if X.shape[1] == 3:
    X = X.transpose()
    Y = Y.transpose()
  #shape of X, Y is 3 * J
  R, t = horn87(X, Y)
  return ((np.dot(R, X) + np.dot(t.reshape(3, 1), np.ones((1, Y.shape[1]))) - Y) ** 2).sum()


if __name__ == '__main__':
  S = np.random.randn(3, 10) 
  tt = np.random.randn(3, 1)
  RR = ortho_group.rvs(dim = 3)
  while np.linalg.det(RR) < 0:
    RR = ortho_group.rvs(dim = 3)
  T = np.dot(tt, np.ones((1, 10))) + np.dot(RR, S)

  print 'tt', tt
  print 'RR', RR, np.dot(RR, RR.transpose(1, 0)), np.linalg.det(RR)

  R, t = horn87(S, T)

  print 't', t
  print 'R', R
