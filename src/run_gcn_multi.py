#!/usr/bin/env python

import numpy as np

import sys,os
import pickle as pk

if len(sys.argv) < 2:
    print('''./run_gcn_multi.py path/to/dataset.pkl "{'layers': 1, 'moment': 1, 'units': 1, 'bias': False, 'activation': 'linear'}"''')
    exit()



EPOCHS = 2000 # total 
EP_SPLIT = 200 # epochs for saving
SAVE_PATH = '../experiments/gcn_grid/'

DATA = pk.load(open(sys.argv[1],'rb'))
params = {'layers': 1, 'moment': 1, 'units': 1, 'bias': False, 'activation': 'linear'}


if len(sys.argv) > 2:
    print('updating parameters...')
    params.update(eval(sys.argv[2]))
    
units = [params['units']] * params['layers']
labels = DATA['moments'][params['moment']]
h = np.ones_like(labels)

Adjacencies = DATA['Adj']
indices = DATA['rand_indices']

import GraphConvNet as gcn
gcn_model = gcn.MultiGCN(input_shape= labels[0].shape, units = units, activation = params['activation'],  
                     dense_kws={'use_bias':params['bias']},
                       GCN_kws={'use_bias':params['bias']},
                        )

for i in range(int(EPOCHS//EP_SPLIT)):
    gcn_model.train([ Adjacencies[indices], h[indices]],[labels[indices]],epochs= EP_SPLIT)
    print('Saving....\n')
    gcn_model.save(SAVE_PATH + 'gcn_model_Lay%d-units%d-moment%d-Act_%s-bias%d.pkl' %(params['layers'], params['units'], 
                        params['moment'], params['activation'], params['bias']))
    
