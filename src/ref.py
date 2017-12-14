edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5], 
         [4, 8], [5, 9], [3, 7], [2, 6]]
totalViewsModelNet = 18
totalViewsShapeNet = 18
nValViews = 18

J = 10
metaDim = 5 + J

eps = 1e-6
imgSize = 224
#Change data dir here
#ShapeNet_dir = '..data/ShapeNet/'
ShapeNet_dir = '/hdd/zxy/data/ShapeNet/'
#ModelNet_dir = '..data/ModelNet/'
ModelNet_dir = '/hdd/zxy/data/ModelNet/'
DCNN_dir = '..data/3DCNN/'
#Redwood_dir = '..data/Redwood_depth/'
Redwood_dir = '/hdd/zxy/data/Redwood_depth/'
RedwoodRGB_dir = '..data/Redwood_RGB/'

ModelNet_version = ''
ShapeNet_version = ''
DCNN_version = ''
Annot_ShapeNet_version = ''
Redwood_version = ''
RedwoodRGB_version = ''
category = 'Chair'
tag = ''
exp_name = '{}{}'.format(category, tag)

