import torchvision.models as models
import ref
import torch
import torch.nn as nn
import os
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def getModel(args):
  # create model
  if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    if args.arch.startswith('resnet'):
      if '18' in args.arch:
        model.fc = nn.Linear(512 * 1, ref.J * 3)
      else :
        model.fc = nn.Linear(512 * 4, ref.J * 3)
      print 'reset classifier'
    if args.arch.startswith('densenet'):
      if '161' in args.arch:
        model.classifier = nn.Linear(2208, ref.J * 3)
      elif '201' in args.arch:
        model.classifier = nn.Linear(1920, ref.J * 3)
      else:
        model.classifier = nn.Linear(1024, ref.J * 3)
    if args.arch.startswith('alex'):
      feature_model = list(model.classifier.children())
      feature_model.pop()
      feature_model.append(nn.Linear(4096, ref.J * 3))
      model.classifier = nn.Sequential(*feature_model)
  else:
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes = ref.J * 3)

  model = torch.nn.DataParallel(model).cuda()
        
  if args.loadModel:
    if os.path.isfile(args.loadModel):
      print("=> loading model '{}'".format(args.loadModel))
      checkpoint = torch.load(args.loadModel)
      model.load_state_dict(checkpoint['state_dict'])
    else:
      raise Exception("=> no model found at '{}'".format(args.loadModel))
  
  return model

