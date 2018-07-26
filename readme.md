# Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency

This repository is the PyTorch implementation for the network presented in:

> Xingyi Zhou, Arjun Karpur, Chuang Gan, Linjie Luo, Qixing Huang, 
> **Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency**
> ECCV 2018([arXiv:1712.05765](https://arxiv.org/abs/1712.05765))

Contact: [zhouxy2017@gmail.com](mailto:zhouxy2017@gmail.com)

## Requirements
- cudnn
- [PyTorch](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 

## Data
- The following datasets are used in this repo. If you use the data provided, please also consider citing them:
  - [ModelNet](http://modelnet.cs.princeton.edu/)
  - [ShapeNet](https://www.shapenet.org/) and keypoint annotation provided in [SyncSpecCNN](https://github.com/ericyi/SyncSpecCNN).
  - [Redwood Dataset](http://redwood-data.org/3dscan/dataset.html?c=chair)
- Download the pre-processing data and annotations [here](https://drive.google.com/open?id=10QGzsukvkeOceqRu8bsfpEaFECRTZCZi), and un-zip them on `data`. 

## Testing
- Download our pre-trained [model](https://drive.google.com/open?id=1nXNPHr8UffI79yT0fBPOy-mTb5iqoYe6) on [Redwood](http://redwood-data.org/3dscan/dataset.html?c=chair) Depth dataset and move it to `models`.
- Run the test.
```
 python main.py -expID demo -loadModel ../models/Redwood.pth.tar -test
```
- Visualize the results.
```
python tools/vis.py ../exp/Chair/demo/img_valTarget ../exp/Chair/demo/valTarget.txt
```

## Training
- Stage1: Train the source model.
```
python main.py -expID Source -epochs 120 -dropLR 90
```

Our results of this stage is provided [here](https://drive.google.com/file/d/1UtlL7moKtNoVGyqWGRn8_c_57dwiqlVm/view?usp=sharing). 

- Stage2: Adapt to the target domain with shape consistency loss.
```
python main.py -expID Redwood -targetDataset Redwood -targetRatio 1 -shapeWeight 1 -loadModel ../models/ModelNet120.tar -LR 0.01
```





