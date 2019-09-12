
import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
sess = tf.InteractiveSession()

from keras.backend import set_session
set_session(sess)

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Add
from keras.layers import Layer, Input, InputSpec, Concatenate, Lambda
from keras.optimizers import Adam, RMSprop
from keras import layers

from keras import backend as K

class EpochHistory(tf.keras.callbacks.Callback):
    
#     def on_train_begin(self, logs={}):
    def __init__(self, metrics=[], get_best_weights = False, verbosity = 10):
        """ metrics: 'acc' to save 'acc' and 'val_acc'; always saves 'loss', 'val_loss'
        
        """
        self.history = {k:[] for k in metrics+['loss','val_loss']}
        self.best_weights = None
        self.best_val_loss = 0
        self.get_best_weights = get_best_weights
        self.verbosity = verbosity
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        for k in self.history:
            self.history[k].append(logs.get(k))
        if self.verbosity and (epoch % self.verbosity) ==0:
            print('Ep: %d\t' %epoch + '\t'.join(['%s: %.4g' %(k,self.history[k][-1]) for k in self.history]))
        if self.get_best_weights and self.history['val_loss'][-1] < self.best_val_loss:
            self.best_val_loss = self.history['val_loss'][-1]
            print('Validation loss improved; getting weights')
#             self.best_weights = self.model.get_weights()
#             if (epoch % 10) ==0 and epoch >1:
#                 print('saving model...')
#                 self.model.save('./graph-gen-test.h5')
                
        
def mat_pow_batch(A,n):
    """batch matrix power"""
    if n ==0:
        return np.float32([np.eye(A.shape[1])])
    M = A
    for _ in range(n-1):
        M = K.tf.matmul(A, M)
    return M.eval()
            
    
#######################


from keras.layers import Dense, InputSpec
import pickle as pk

class GCN(Dense):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(GCN, self).__init__(**kwargs)
        # gotta redefine input_spec b/c we have two inputs whiule Dense has one
        self.input_spec = [InputSpec(min_ndim=2),InputSpec(min_ndim=2)]
        self.__doc__ = """ Same arguments as dense, except that it is called with [M,h] as input
        Dense Doc:
        """+ super(GCN, self).__doc__
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        M_shape, h_shape = input_shape
        super(GCN, self).build(h_shape)  # Be sure to call this at the end
        # Dense.build redefines input_spec; change it to avoid error when calling with two inputs [M,h]
        self.input_spec = [InputSpec(min_ndim=2),self.input_spec]
    
    def call(self, inputs):
        M,h = inputs
        Mh = K.batch_dot(M,h)
        return super(GCN, self).call(Mh)
    
    def compute_output_shape(self, input_shape):
        M_shape, h_shape = input_shape
        return super(GCN, self).compute_output_shape(h_shape)
    
class Graph_Operators(Layer):
    """Takes a batch of adjacency matrices and outputs [A, D^{-1}A, D^{-1/2}AD^{-1/2}]"""
    eps = 1e-8 # to avoid nan for inverse degree. works for float32
    def __init__(self, **kwargs):
        super(Graph_Operators, self).__init__(**kwargs)
        
    def call(self, A):
        D = K.tf.reduce_sum(A, axis = -1, keepdims = True)
        # To make D^{-1}A, we can simply multiply 1/D from the left, but we have to add a new axis 
        Dinv = K.tf.to_float(D > 0) / (D + self.eps)
        DA = Dinv * A 
        # for D^{-1/2}AD^{-1/2}, the following pointwise operation works
        d2 = K.tf.sqrt(Dinv) # D^{-1/2}
        DAD =  d2 * A * d2[:,np.newaxis,:,0]
        
        #out = OrderedDict([('A', A), ('DA', DA), ('DAD', DAD)])
        out = [A,DA,DAD]
        self.op_names = ['A', 'DA', 'DAD']
        self.out_len = len(out)
        return out  
                
    def compute_output_shape(self, input_shape):
        return [input_shape]*self.out_len
    

# make GCN list without names , to avoid error in multilayer case. 
# make GCN list able to take tensor with 3 channels and spread it between the ops

def GCN_List(gr_ops, inputs, units=1, names = None, **kwargs):
    """ gr_ops: output of Graph_Operators
    kwargs passed to GCN
    names: used to name operators
    """
    if isinstance(inputs, K.tf.Tensor):
        inputs = Lambda(lambda x: [x[...,i:i+1] for i in range(x.shape.as_list()[-1])])(inputs) 
    if names == None:
        names = [None for i in range(len(gr_ops))]
    out = [ GCN(units = units, name = nam, **kwargs)([g, inp]) for g, nam, inp in zip(gr_ops,names, inputs) ]
    #return Concatenate()(out)
    return out

# Add GCN module

class GCN_module:
    def __init__(self, graph_operators, inputs, units = 1, GCN_kws = {}, dense_kws = {}):
        self.GCN_keywords = GCN_kws
        self.Dense_Keywords = dense_kws
        self.gcn_units =  GCN_List(graph_operators, inputs, units = units, **GCN_kws)
        self.dense_layer = Dense(len(graph_operators), **dense_kws) 
        self.output = self.dense_layer(Concatenate()(self.gcn_units))
    
    def get_weights(self):
        return {'dense': self.dense_layer.get_weights(),
            'gcn': [g._keras_history[0].get_weights() for g in self.gcn_units]} # g is a tensor, but links to a keras layer 
    
    def get_weighted_contributions(self):
        """ returns a 3x3 matrix showing the contibution of A, DA, ADA weights in the outputs of the layer """
        w = self.get_weights()
        wd = w['dense']
        wg = w['gcn']
        # num gcn weights
        u = wg[0][0].shape[1]
        
        # concat weights of gcn units, as done for input of Dense
        wc = np.concatenate([i[0] for i in wg], axis = -1)
        
        # weight the rows of the Dense weights with the gcn weights
        w1 = (wd[0].T*wc)
        
        # sum the columns for each gcn unit
        ws = np.zeros((len(w1), len(w1)))
        for i in range(0,w1.shape[1], u):
            ws[:,i//u] = w1[:,i:i+u].sum(1)
            
        return ws.T
        
        
        
        
######

class MultiGCN():
    def __init__(self, input_shape, output_shape='same', units=[1], activation = None, skip= True, 
                 GCN_kws={}, dense_kws={}, final_kws={},verbosity=50, from_file = None):
        """
        input_shape: shape of the node attribute matrix H (nodes, attribs); 
            shape of adjacency matrix is inferred from H 
        units: list of output weight channels for GCN layers
        activation: used both for GCN and the immediately following Dense layer. 
        skip: whether to concatenate the out put of all GCN layers to the input of the final dense layer 
        
        """
        
        # assert input_shape[0]==output_shape[0], 'Output shape not consistent with GCN: output_shape[0] must be number of nodes.'
        self.input_shape = input_shape
        self.output_shape = (input_shape if output_shape == 'same' else output_shape)
        self.units = units
        self.activation = activation
        self.skip = skip
        self.GCN_kws = GCN_kws
        self.dense_kws = dense_kws
        if activation != None:
            self.GCN_kws.update({'activation':activation})
            self.dense_kws.update({'activation':activation})
            
        self.final_kws = {'units': self.output_shape[-1]}
        self.final_kws.update(final_kws)
        
        ##### keep all params for saving
        self._params = {
            'units': self.units,
            'activation': self.activation,
            'skip': self.skip,
            'GCN_kws': self.GCN_kws,
            'dense_kws': self.dense_kws,
        }
        
        self._verbosity = verbosity
        
        if from_file:
            f = pk.load(open(from_file, 'rb'),)
            self._params = f['params']
            self._reset_params()
            self._loaded_weights = f['weights']
        
        self.build_model()
        if from_file: self.model.set_weights(self._loaded_weights)
        
    def _reset_params(self):
        for k in self._params:
            setattr(self, k, self._params[k])
            
    def save(self, fname):
        d = {'params':self._params, 'weights':self.model.get_weights(), 'history':self.history.history }
        pk.dump(d, open(fname,'wb'))
        
        
    def build_model(self):
        Adj_shape = tuple(2*[self.input_shape[0]])
        
        Adjacency, H = Input(shape = Adj_shape), Input(shape= self.input_shape)

        gr_ops_ = Graph_Operators()
        gr_ops = gr_ops_(Adjacency)
        
        # Make GCN modules
        self.gcn_modules = []
        GCN_kws = self.GCN_kws.copy()
        for i,u in enumerate(self.units):
            inputs = ( self.gcn_modules[-1].output if i>0 else len(gr_ops)*[H] )
            GCN_kws['names']=['%s_%d' %(s,i)  for s in gr_ops_.op_names]
            self.gcn_modules += [GCN_module( gr_ops, inputs, units = u, GCN_kws = GCN_kws, dense_kws = self.dense_kws )]
                
        if self.skip and len(self.gcn_modules)>1:
            final_layer_input = Concatenate()([gcn.output for gcn in self.gcn_modules])
        else:
            final_layer_input = self.gcn_modules[-1].output 
            
        #final_layer = Dense(units = self.output_shape[-1], **self.final_kws)( final_layer_input )
        final_layer = Dense(**self.final_kws)( final_layer_input )


        self.model = Model(inputs = [Adjacency, H] , outputs = final_layer )

        self.model.compile(loss='mse', optimizer='adam') #'rmsprop')
        # self.model.summary()
        
        self.history = EpochHistory(verbosity=self._verbosity)
        
        
    def train(self, inputs, labels, epochs = 600, validation_split=0.2,verbose=False,**kws):
        """inputs = [A,H], labels = Y"""
        history = self.model.fit(inputs,labels, validation_split=validation_split, epochs= epochs, 
                        callbacks=[self.history], verbose=verbose, **kws)
        
    def plot_loss(self):
        for s in ['loss','val_loss']:
            y = self.history.history[s]
            plt.plot(np.arange(len(y)),y, label = s)
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('MSE Loss', size = 14)

        
    def get_propagated_gcn_contrib(self):
        """Propagate gcn"""
        ws = []
        for g in self.gcn_modules:
            ws +=[g.get_weighted_contributions()]

        # get weights of last dense layer
        wd,bd = self.model.layers[-1].get_weights()
        w = [ws[0]]
        for v in ws[1:]:
            # make list of all gcn layer outputs
            w += [w[-1].dot(v)]

        # make list of output weights from propagated weights and last layer's weights. 
        l = len(w[0])
        if self.skip:
            wf = [w[i].dot(wd[i*l:(i+1)*l]) for i in range(len(w))]
        else:
            wf = [w[-1].dot(wd)]
        return wf



    def plot_op_contrib(self):
        """
        !!!! There is a full matrix of contirbutions... can't just attribute them to A^n,...! 
        
        plot the contribution of each operator to the output.
        (Uses self.get_propagated_gcn_contrib())
        """
        print("Note: This only uses weight matrices with linear activation and ignores biases!")
        wf = self.get_propagated_gcn_contrib()
        wi = np.concatenate(wf).ravel()
        num_ops = len(wi)//len(wf[0])
        plt.bar(range(len(wi)), wi)
        # yscale('log')
        labels = [r'$A$', r'$\hat{A}$', r'$\hat{A}_{s}$']
        for i in range(1,num_ops):
            labels += [r'$A^%d$' %(i+1), r'$\hat{A}^%d$' %(i+1), r'$\hat{A}_{s}^%d$' %(i+1)]
        plt.xticks(range(len(labels)), labels, size = 14, rotation = 0)

        plt.ylabel('weight contrib.',size = 14)

    
