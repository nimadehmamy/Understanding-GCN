#!/usr/bin/env python

import numpy as np

import sys,os
import pickle as pk

if len(sys.argv) < 2:
    print('''./run_gcn_multi.py path/to/dataset.pkl [moment order (1,2,3)] ["{MultiGCN keywords}"]''')
    exit()



EPOCHS = 4000 if len(sys.argv) < 5 else int(sys.argv[4])  # total 
EP_SPLIT = 200 # epochs for saving
SAVE_PATH = '../experiments/gcn_grid/'

DATA = pk.load(open(sys.argv[1],'rb'))
MOMENT = int(sys.argv[2])

params = {
    'units':[1],
    'activation':'linear',
    'skip':True,
}

if len(sys.argv) > 2:
    print('updating parameters...')
    params.update(eval(sys.argv[3]))
    
labels = DATA['moments'][MOMENT]
h = np.ones_like(labels)

indices = DATA['rand_indices']

import GraphConvNet as gcn
gcn_model = gcn.MultiGCN(input_shape= labels[0].shape, **params )

for i in range(int(EPOCHS//EP_SPLIT)):
    gcn_model.train([ DATA['Adj'][indices], h[indices]],[labels[indices]],epochs = EP_SPLIT)
    print('%d) Saving....\n' %i)
    gcn_model.save(SAVE_PATH + 'gcn_model-units%s-moment%d-Act_%s-skip%d.pkl' %(params['units'], 
                        MOMENT, params['activation'], params['skip']))
    
