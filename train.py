import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import shutil
import networkx as nx
import pickle as pkl
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from tfdeterminism import patch
from model.GraphSleepNet import build_Model
from model.Utils import *
from model.DataGenerator import *
from tensorflow.python.client import device_lib
import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#seed = 20200220
patch()
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
def summarize_keras_trainable_variables(model, message):
  s = sum(map(lambda x: x.sum(), model.get_weights()))
  print("summary of trainable variables %s: %.13f" % (message, s))
  return s

seed = 2022
set_random_seeds(seed)

Path, cfgTrain, cfgModel = ReadConfig('./config/run.config')

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Use GPU #"+'0')

# [train] parameters
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lr_decay   = float(cfgTrain["lr_decay"])
subject_id = int(cfgTrain["subject_id"])
# [model] parameters
dense_size            = np.array(str.split(cfgModel["Globaldense"],','),dtype=int)
conf_adj              = cfgModel["adj_matrix"]
GLalpha               = float(cfgModel["GLalpha"])
num_of_chev_filters   = int(cfgModel["cheb_filters"])
num_of_time_filters   = int(cfgModel["time_filters"])
time_conv_strides     = int(cfgModel["time_conv_strides"])
time_conv_kernel      = int(cfgModel["time_conv_kernel"])
num_block             = int(cfgModel["num_block"])
cheb_k                = int(cfgModel["cheb_k"])
l1                    = float(cfgModel["l1"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])


# check optimizer（opt）
if optimizer=="adam":
    opt = keras.optimizers.Adam(lr=learn_rate,decay=lr_decay)
elif optimizer=="RMSprop":
    opt = keras.optimizers.RMSprop(lr=learn_rate,decay=lr_decay)
elif optimizer=="SGD":
    opt = keras.optimizers.SGD(lr=learn_rate,decay=lr_decay)
elif optimizer=="adamw":
    opt = keras.optimizers.Adamax(lr=learn_rate, decay=lr_decay)
else:
    assert False,'Config: check optimizer'

# set l1、l2（regularizer）
if l1!=0 and l2!=0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1!=0 and l2==0:
    regularizer = keras.regularizers.l1(l1)
elif l1==0 and l2!=0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None
    
# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile('config/run.config', Path['Save'] + "last.config")


# # 2. Read data and process data
idx = subject_id
print(idx)

cheb_polynomials = None

# Model training (cross validation)

# 4-fold cross validation
all_scores = []
for i in range(0,1):
    subject = [1,2,3,4,5,6,7,8,9]
    isfirst = True
    for idx in subject:
        path = r'.\test\22_channels_' + str(idx) + '.npz'
        ReadList = np.load(path, allow_pickle=True)
        Fold_Data = ReadList['Fold_Data']
        Fold_Label = ReadList['Fold_Label']
        print(Fold_Data.shape)
        print("Read data successfully")
        Fold_Data = AddContext(Fold_Data, context)
        Fold_Label = AddContext(Fold_Label, context, label=True)
        print('Context added successfully.')
        adj = np.load(r'.\test\\de_adj_' + str(idx) + '.npz')
        Fold_Data = tf.constant(np.array(Fold_Data))
        Fold_Data = tf.tile(Fold_Data, [1, 1, 1, 1, 2])
        with tf.Session() as sess:
            Fold_Data = Fold_Data.eval()
        print(Fold_Data.shape)
        train_adj = adj['train_adj']
        valid_adj = adj['valid_adj']
        train_adj = np.expand_dims(train_adj, axis=1)
        valid_adj = np.expand_dims(valid_adj, axis=1)
        print(valid_adj.shape)
        DataGenerator = kFoldGenerator(fold, Fold_Data, Fold_Label)
        print('ok')
        train_data,train_targets,val_data,val_targets = DataGenerator.getFold(i)
        print(train_data.shape)
        train_data = np.concatenate([train_data, train_adj], axis=1)
        val_data = np.concatenate([val_data, valid_adj], axis=1)
        if isfirst:
            all_train_data = train_data
            all_train_label = train_targets
            all_val_data = val_data
            all_val_label = val_targets
            isfirst = False
        else:
            all_train_data = np.concatenate([all_train_data,train_data],axis=0)
            all_train_label = np.concatenate([all_train_label,train_targets],axis=0)
            all_val_data  = np.concatenate([all_val_data,val_data],axis=0)
            all_val_label = np.concatenate([all_val_label,val_targets],axis=0)
    print(all_train_data.shape)

    index = [j for j in range(len(all_train_data))]
    random.shuffle(index)
    all_val_data = all_val_data[index]
    all_val_label = all_val_label[index]
    all_train_data = all_train_data[index]
    all_train_label = all_train_label[index]

    #print(train_data.shape)
    # build model
    sample_shape = (2, all_train_data.shape[2], all_train_data.shape[3])
    model=build_Model(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel,
                      sample_shape, num_block, dense_size, opt, conf_adj =='GL',GLalpha, regularizer, dropout)
    summarize_keras_trainable_variables(model, "before training")
    if i==0:
        model.summary()
    # train
    history = model.fit(
        x = all_train_data[int(648):],
        y = all_train_label[int(648):],
        epochs = num_epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = (all_train_data[0:int(648)], all_train_label[0:int(648)]),
        callbacks=[keras.callbacks.ModelCheckpoint('Best_model_'+str(i)+'.h5',
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True, 
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   period=1)], verbose = 1)
    model.save('Final_model_'+str(i)+'.h5')
    Fold_Num = [288*9 for _ in range(9)]
    # Save training information
    if i==0:
        fit_acc=np.array(history.history['acc'])*Fold_Num[i]
        fit_loss=np.array(history.history['loss'])*Fold_Num[i]
        fit_val_loss=np.array(history.history['val_loss'])*Fold_Num[i]
        fit_val_acc=np.array(history.history['val_acc'])*Fold_Num[i]
    else:
        fit_acc=fit_acc+np.array(history.history['acc'])*Fold_Num[i]
        fit_loss=fit_loss+np.array(history.history['loss'])*Fold_Num[i]
        fit_val_loss=fit_val_loss+np.array(history.history['val_loss'])*Fold_Num[i]
        fit_val_acc=fit_val_acc+np.array(history.history['val_acc'])*Fold_Num[i]

    model.load_weights(r'C:\Users\ly\Downloads\GraphSleepNet-master\GraphSleepNet-master'+'\\' +'Best_model_'+str(i)+'.h5')
    val_mse, val_acc = model.evaluate(all_val_data, all_val_label, verbose=1)
    print('Evaluate', val_acc)
    all_scores.append(val_acc)
    
    # Predict
    predicts = model.predict(all_val_data)
    AllPred_temp = np.argmax(predicts, axis=1)
    AllTrue_temp = np.argmax(all_val_label, axis=1)
    PrintScore(AllTrue_temp, AllPred_temp)
    print(128*'_')
del model, train_data, train_targets, val_data, val_targets
