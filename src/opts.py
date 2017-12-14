import argparse
import os
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser(description='3D Keypoint')
  
  def init(self):
    self.parser.add_argument('-expID', default = 'defult', 
                               help='path to save')
    self.parser.add_argument('-targetDataset', default='Redwood', type=str, 
                               help='Redwood | ShapeNet | RedwoodRGB | 3DCNN')
    self.parser.add_argument('-test', action = 'store_true', help='test')
    self.parser.add_argument('-DEBUG', default = 0, type = int, help='debug level')
    self.parser.add_argument('-arch', default='resnet50')
    self.parser.add_argument('-workers', default=4, type=int, metavar='N', help='#data loading workers (default: 4)')
    self.parser.add_argument('-epochs', default=30, type=int, help='number of total epochs to run')
    self.parser.add_argument('-dropLR', default=20, type=int, metavar='N', help='# total epochs to drop LR')
    self.parser.add_argument('-batchSize', default=64, type=int, help='mini-batch size (default: 64)')
    self.parser.add_argument('-LR', default=0.1, type=float, help='initial learning rate')
    self.parser.add_argument('-momentum', default=0.9, type=float, help='momentum')
    self.parser.add_argument('-weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    self.parser.add_argument('-loadModel', default='', type=str, help='path to loadmodel (default: none)')
    self.parser.add_argument('-pretrained', action='store_true', help='use pre-trained model')
    self.parser.add_argument('-shapeNetFullTest', action='store_true', help='shapeNetFullTest')

    self.parser.add_argument('-intervalUpdateM', default=5, type=int, help='update M')
    self.parser.add_argument('-saveVis', action = 'store_true', help='')
    
    self.parser.add_argument('-totalTargetIm', default=1, type=float, help='training target image num')
    self.parser.add_argument('-targetRatio', default=0, type=float, help='ratio of training target image num')
    self.parser.add_argument('-shapeWeight', default=0, type=float, help='shape consistancy weight')
    self.parser.add_argument('-nViews', default=8, type=int, help='group for unsupervised constraint')
    self.parser.add_argument('-AVG', action = 'store_true', help='')
    
    self.parser.add_argument('-shapenetAnnot', default='No', type=str, help='No | All')
    
    self.parser.add_argument('-sampleSource', default=1, type = int, help='sample source domain')
    self.parser.add_argument('-lamb', default=1, type = float, help='lamb')
    self.parser.add_argument('-mu', default=0.1, type = float, help='mu')
    
    
  def parse(self):
    self.init()
    self.args = self.parser.parse_args()
    if self.args.test and not self.args.expID.endswith('TEST'):
      self.args.expID = self.args.expID + 'TEST'
    if not os.path.exists('../exp/{}/'.format(ref.exp_name)):
      os.mkdir('../exp/{}/'.format(ref.exp_name))
    self.args.save_path = '../exp/{}/'.format(ref.exp_name) + self.args.expID
    
    self.args.batchSize = self.args.batchSize / self.args.nViews
    print '# model per batch: {}, # views: {} '.format(self.args.batchSize, self.args.nViews) 
    
    if not os.path.exists(self.args.save_path):
      os.mkdir(self.args.save_path)
      if self.args.test:
        os.mkdir(self.args.save_path + '/img_train')
        os.mkdir(self.args.save_path + '/img_valSource')
        os.mkdir(self.args.save_path + '/img_valTarget')
    return self.args
