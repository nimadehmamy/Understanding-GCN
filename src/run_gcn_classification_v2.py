#!/usr/bin/env python

import numpy as np

import sys,os
import pickle as pk

if len(sys.argv) < 2:
    print('''./run_gcn_multi.py path/to/dataset.pkl ["{MultiGCN keywords}"]''')
    exit()

SAVE_PATH = '../experiments/gcn_classification_grid/' if len(sys.argv) < 4 else sys.argv[3]
os.makedirs(SAVE_PATH, exist_ok=True)

EPOCHS = 200 if len(sys.argv) < 5 else int(sys.argv[4])  # total 
EP_SPLIT = 50 # epochs for saving

DATA = pk.load(open(sys.argv[1],'rb'))

params = {
    'units':[1],
    'activation':'linear',
    'skip':True,
}

if len(sys.argv) > 2:
    print('updating parameters...')
    params.update(eval(sys.argv[2]))
    
labels = DATA['labels']
Adjacencies = DATA['Adj']
h = np.ones(Adjacencies.shape[:2]+(1,))
indices = DATA['rand_indices']


import GraphConvNet as gcn
gcn_model = gcn.MultiGCN(input_shape = h[0].shape, **params )

# use leaky ReLU on the last layer
ac = gcn.layers.LeakyReLU(alpha = 0.3)(gcn_model.model.output)
# classification_layer = gcn.Dense(len(labels[0]), activation='softmax')( gcn.layers.Flatten()( ac ) )
# classification_layer = gcn.Dense(len(labels[0]), activation='softmax')( gcn.layers.Flatten()( gcn_model.model.output ) )

den1 = gcn.Dense(len(labels[0]), activation='softmax')( ac ) # has dims N x out channels must sum over N
# make layer to average over nodes
Avg_Nodes = gcn.layers.Lambda(lambda x: gcn.K.mean(x,axis = 1))
classification_layer = Avg_Nodes(den1)

model = gcn.Model(inputs = gcn_model.model.inputs, outputs = [classification_layer] )
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.summary()

gcn_model.history = gcn.EpochHistory(metrics=['acc','val_acc']) 
gcn_model.model = model

g_nam = '-gcn[units%s-%s]' %(params['units'], '-'.join( ['%s_%s' %(v) for v in (gcn_model.GCN_kws.items())] ))
final_nam = '-final[%s]' %('-'.join( ['%s_%s' %(v) for v in (gcn_model.final_kws.items())] ))

for i in range(int(EPOCHS//EP_SPLIT)):
    gcn_model.train([ DATA['Adj'][indices], h[indices]],[labels[indices]],epochs = EP_SPLIT)
    print('%d) Saving....\n' %i)
    gcn_model.save(os.path.join(SAVE_PATH , 'gcn_model-N%d%s%s-skip%d.pkl' %(DATA['Adj'].shape[1],g_nam, final_nam, gcn_model.skip)))
    
