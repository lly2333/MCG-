import sys
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import regularizers
from keras import models
from keras.layers import Layer, Concatenate
from keras.layers.core import Lambda
from keras.initializers import RandomUniform, glorot_normal
import random
import os
# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
# 
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)
# 
#     5: 5 sleep stages
seed = 20200220
def set_random_seeds(seed):
    """Set seeds for python random module numpy.random and torch.

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
set_random_seeds(seed)
def compute_cosdis(data):
    adj_matrix =np.zeros((22,22))
    matrix = tf.Session().run(data)
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            tmp = np.dot(matrix[i], matrix[j]) / (np.linalg.norm(matrix[i]) *np.linalg.norm(matrix[j]))
            adj_matrix[i][j] = tmp
            adj_matrix[j][i] = tmp
            #print(tmp)
    return tf.convert_to_tensor(adj_matrix)

class TemporalAttention(Layer):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                      shape=(num_of_vertices, 1),
                                      #initializer='uniform',
                                      initializer = RandomUniform(seed=seed),
                                      trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                      shape=(num_of_features, num_of_vertices),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.U_3 = self.add_weight(name='U_3',
                                      shape=(num_of_features, ),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                      shape=(1, num_of_timesteps, num_of_timesteps),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.V_e = self.add_weight(name='V_e',
                                      shape=(num_of_timesteps, num_of_timesteps),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, T, V)
        lhs=K.dot(tf.transpose(x,perm=[0,1,3,2]), self.U_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_timesteps,num_of_features])
        lhs = K.dot(lhs, self.U_2)
        
        # shape of rhs is (batch_size, V, T)
        rhs = K.dot(self.U_3, tf.transpose(x,perm=[2,0,3,1])) # K.dot((F),(V,batch_size,F,T))=(V,batch_size,T)
        rhs=tf.transpose(rhs,perm=[1,0,2])
        
        # shape of product is (batch_size, T, T)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_e, tf.transpose(K.sigmoid(product + self.b_e),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[1])


class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                      shape=(num_of_timesteps, 1),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                      shape=(num_of_features, num_of_timesteps),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                      shape=(num_of_features, ),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                      shape=(1, num_of_vertices, num_of_vertices),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                      shape=(num_of_vertices, num_of_vertices),
                                      #initializer='uniform',
                                      initializer=RandomUniform(seed=seed),
                                      trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, V, T)
        lhs=K.dot(tf.transpose(x,perm=[0,2,3,1]), self.W_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_vertices,num_of_features])
        lhs = K.dot(lhs, self.W_2)
        
        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x,perm=[1,0,3,2])) # K.dot((F),(V,batch_size,F,T))=(V,batch_size,T)
        rhs=tf.transpose(rhs,perm=[1,0,2])
        
        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2],input_shape[2])


def diff_loss(diff, S):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape)==4:
        # batch input
        return K.mean(K.sum(K.sum(diff**2,axis=3)*S, axis=(1,2)))
    else:
        return K.sum(K.sum(diff**2,axis=2)*S)


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * K.sum(K.mean(S**2,axis=0))
    else:
        return Falpha * K.sum(S**2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_ of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 #initializer='uniform',
                                 initializer=RandomUniform(seed=seed),
                                 trainable=True)
        # add loss L_{graph_learning} in the layer
        self.add_loss(F_norm_loss(self.S,self.alpha))
        self.add_loss(diff_loss(self.diff,self.S))
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
        print(N)
        # shape: (N,V,F) use the current slice (middle one slice)
        x = x[:,int(x.shape[1])//2,:,:]
        # shape: (N,V,V,F)
        diff = tf.transpose(tf.transpose(tf.broadcast_to(x,[V,N,V,F]), perm=[2,1,0,3])-x, perm=[1,0,2,3])
        # shape: (N,V,V)
        tmpS = K.exp(K.relu(K.reshape(K.dot(K.abs(diff), self.a), [N,V,V])))
        # normalization
        S = tmpS / K.sum(tmpS,axis=1,keepdims=True)
        
        self.diff = diff
        self.S = S
        return S

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices, num_of_vertices)
        return (input_shape[0],input_shape[2],input_shape[2])


class Graph_Learn1(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_ of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(Graph_Learn1, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        super(Graph_Learn1, self).build(input_shape)

    def call(self, x):
        # Input:  [N, timesteps, vertices, features]
        x, S = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x)
        #S = Lambda(tf.squeeze, arguments={'axis': 1})(S)
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
        isfirst1 = True
        for i in range(int(x.shape[2])):
            isfirst = True
            for j in range(int(x.shape[3])):
                s = tf.reduce_sum(tf.multiply(x[:,0,i,:], x[:,0,j,:]), 1) / (tf.norm(x[:,0,i,:],ord=2) *tf.norm(x[:,0,j,:],ord=2))
                s = Lambda(tf.expand_dims, arguments={'dim': 1})(s)
                #print(s.shape)
                if isfirst:
                    s1 = s
                    isfirst = False
                else:
                    s1 = Concatenate(axis=1)([s1, s])
                    #print(s1.shape)
            s1 = Lambda(tf.expand_dims, arguments={'dim': 1})(s1)
            if isfirst1:
                S1 = s1
                isfirst1 = False
            else:
                S1 = Concatenate(axis=1)([S1,s1])
        out = S1
        return out

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[2], input_shape[2])

class cheb_conv_with_SAt_GL(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_SAt_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape,S_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     #initializer='uniform',    #均匀分布初始化
                                     initializer=RandomUniform(seed=seed),
                                     trainable=True)
        super(cheb_conv_with_SAt_GL, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt,S]
        assert isinstance(x, list)
        assert len(x)==3,'cheb_conv_with_SAt_GL: number of input error'
        x, spatial_attention, W = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        #Calculating Chebyshev polynomials
        D = tf.matrix_diag(K.sum(W,axis=1))
        L = D - W
        '''
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
            L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        '''
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        
        #Graph Convolution
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape of x is (batch_size, V, F')
            output = K.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)


class cheb_conv_with_SAt_static(Layer):
    '''
    K-order chebyshev graph convolution with static graph structure
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, cheb_polynomials, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        super(cheb_conv_with_SAt_static, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_SAt_static, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt]
        assert isinstance(x, list)
        assert len(x)==2,'cheb_conv_with_SAt_static: number of input error'
        x, spatial_attention = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape is (batch_size, V, F')
            output = K.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = self.cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)


def reshape_dot(x):
    '''
    Apply temporal attention to x
    Input:  [x, TAtt]
    '''
    x, temporal_At = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x,perm=[0,2,3,1]),(tf.shape(x)[0], -1, tf.shape(x)[1])), 
            temporal_At
        ),
        [-1, x.shape[1],x.shape[2],x.shape[3]]
    )


def LayerNorm(x):
    '''
    Apply relu and layer normalization
    '''
    x_residual, time_conv_output = x
    relu_x = K.relu(x_residual + time_conv_output)
    ln = tf.contrib.layers.layer_norm(relu_x, begin_norm_axis=3)
    return ln
def Scompute_mulinfo(data,chans):
    adj_matrix = np.zeros((chans,chans))
    #print(data.shape)
    matrix = data.cpu().detach().numpy()
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            size = matrix[i].shape[-1]
            px = np.histogram(matrix[i], 256, (0, 255))[0] / size
            py = np.histogram(matrix[j], 256, (0, 255))[0] / size
            hx = - np.sum(px * np.log(px + 1e-8))
            hy = - np.sum(py * np.log(py + 1e-8))

            hxy = np.histogram2d(matrix[i], matrix[j], 256, [[0, 255], [0, 255]])[0]
            hxy /= (1.0 * size)
            hxy = - np.sum(hxy * np.log(hxy + 1e-8))

            r = hx + hy - hxy
            adj_matrix[i][j] = r
            adj_matrix[j][i] = r
    #return torch.tensor(adj_matrix, dtype=torch.float32)
    return adj_matrix

def Graph_Block(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, useGL, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input
    '''
        
    # TemporalAttention 
    # output shape is (batch_size, T, T)
    #print(x.shape)
    X, S = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': 2})(x)
    #x, S = tf.split(x, 2, axis=1)
    S = Lambda(tf.squeeze, arguments={'axis': 1})(S)
    #X = x
    temporal_At = TemporalAttention()(X)
    x_TAt = Lambda(reshape_dot,name='reshape_dot'+str(i))([X,temporal_At])

    # SpatialAttention
    # output shape is (batch_size, V, V)
    spatial_At = SpatialAttention()(x_TAt)
    
    # Graph Convolution with spatial attention
    # output shape is (batch_size, T, V, F)
    if useGL:
        # use adaptive Graph Learn
        #print(S[N].shape)
        #print(x[N,:,:,:])
        #X, S = tf.split(x, 2, axis=1)
        spatial_gcn = cheb_conv_with_SAt_GL(num_of_filters=num_of_chev_filters, k=k)([X, spatial_At, S])
        S = Graph_Learn1(alpha=GLalpha)(x)
        #S[N, :, :] = Lambda(compute_cosdis)(x[N, 0, :, :])
    else:
        # use fix graph structure
        spatial_gcn = cheb_conv_with_SAt_static(num_of_filters=num_of_chev_filters, k=k, cheb_polynomials=cheb_polynomials)([x, spatial_At])
    
    # Temporal Convolution
    # output shape is (batch_size, T, V, F')
    time_conv_output = layers.Conv2D(
        filters = num_of_time_filters, 
        kernel_size = (time_conv_kernel, 1),
        kernel_initializer=glorot_normal(seed=seed),
        padding = 'same', 
        strides = (time_conv_strides, 1)
    )(spatial_gcn)

    # residual shortcut
    x_residual = layers.Conv2D(
        filters = num_of_time_filters, 
        kernel_size = (1, 1),
        kernel_initializer=glorot_normal(seed=seed),
        strides = (1, time_conv_strides)
    )(X)
    
    # LayerNorm
    end_output = Lambda(LayerNorm,
                        name='layer_norm'+str(i))([x_residual,time_conv_output])
    S = Lambda(tf.expand_dims, arguments={'dim': 1})(S)
    end_output = Concatenate(axis=1)([end_output, S])
    print(end_output.shape)
    return end_output


def build_Model(k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel,
                sample_shape, num_block, dense_size, opt, useGL, GLalpha, regularizer, dropout):
    
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input-Data')
    # GraphSleepBlock
    block_out = Graph_Block(data_layer,k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel,useGL,GLalpha)
    #print(block_out.shape)
    for i in range(1, num_block):
        block_out = Graph_Block(block_out,k,num_of_chev_filters,num_of_time_filters,1,cheb_polynomials,time_conv_kernel,useGL,GLalpha,i)
    # Global dense layer
    block_out = layers.Flatten()(block_out)
    for size in dense_size:
        block_out=layers.Dense(size,kernel_initializer=glorot_normal(seed=seed))(block_out)
    if dropout!=0:
        block_out = layers.Dropout(dropout,seed=seed)(block_out)
    softmax = layers.Dense(4,activation='softmax',kernel_regularizer=regularizer,kernel_initializer=glorot_normal(seed=seed))(block_out)
    model = models.Model(inputs = data_layer, outputs = softmax)
    
    model.compile(
        optimizer= opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model


def build_GraphSleepNet_test():
    
    # an example to test
    cheb_k = 4
    num_of_chev_filters = 10
    num_of_time_filters = 10
    time_conv_strides = 1
    time_conv_kernel = 3
    dense_size=np.array([64,32])
    cheb_polynomials = [np.random.rand(26,26),np.random.rand(26,26),np.random.rand(26,26)]

    opt='adam'

    model=build_GraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters,time_conv_strides, cheb_polynomials, time_conv_kernel, 
                      sample_shape=(5,22,9),num_block=1, dense_size=dense_size, opt=opt, useGL=False,
                      GLalpha=0.0001, regularizer=None, dropout = 0.0)
    model.summary()
    model.save('GraphSleepNet_build_test.h5')
    print("save ok")
    return model


# build_GraphSleepNet_test()

