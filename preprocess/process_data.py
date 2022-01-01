import scipy.io as sio
import random
import numpy as np
from DE_PSD import *
from sklearn.preprocessing import normalize
import sklearn.metrics as skm

def ReadData(idx,pathF):
    '''
    Read DE or PSD from XXXX_psd_de.mat
    '''
    read_mat=sio.loadmat(pathF+str(idx)+'_psd_de.mat')
    #read_data_psd=read_mat['psd']
    read_data_de=read_mat['de']
    #print(read_data_de.shape,end=' ')
    return read_data_de

def ReadLabel(filename,pathL):
    '''
    Read label from XXXX-Label.mat
    '''
    read_mat=sio.loadmat(pathL+filename+'\\Label.mat')
    Label_lists=read_mat['label']
    #print(Label_lists.shape,end=' ')
    return Label_lists

def transY_onehot(label, num_class):

    onehotlabel = np.zeros((len(label),num_class))
    for i in range(len(label)):
        tmp = np.zeros((num_class))
        tmp[int(label[i])-1] = 1
        onehotlabel[i] = tmp
    return onehotlabel

def Prepare_Data(idx, path, s_type, norm=True):
    Out_Data=[]
    Out_Label=[]
    i = 0
    dataset = np.load(path+str(idx)+'.npz')
    #dataset = np.load(r'C:\\Users\ly\\Desktop\\CHI_2a\\test\\trial_features_' + str(idx) + '.npz')
    FoldData = dataset['all_' + s_type]
    FoldLabel = dataset['all_label']
    FoldLabel = transY_onehot(FoldLabel, 4)
    while i < 8:
        Out_Data.append(FoldData[i*72:(i+1)*72,:,:])
        Out_Label.append(FoldLabel[i*72:(i+1)*72])
        i = i + 1
    All_Data = FoldData
    All_Label = FoldLabel
    # Data standardization
    if norm:
        mean = All_Data.mean(axis=0)
        std = All_Data.std(axis=0)
        All_Data -= mean
        All_Data /= std
        for i in range(8):
            Out_Data[i] -= mean
            Out_Data[i] /= std
    print('All_Data:  ', All_Data.shape)
    print('All_Label: ', All_Label.shape)
    return {
        'Fold_Data':  Out_Data,
        'Fold_Label': Out_Label
        }

if __name__ == "__main__":
    # define the path to load and save
    for idx in range(1,10):
        s_type = 'de'
        ReadList = Prepare_Data(idx,'.\\trial_features_',s_type)
        np.savez(
            '.\\22_channels_' + str(idx)+'.npz',
            Fold_Data = ReadList['Fold_Data'],
            Fold_Label = ReadList['Fold_Label']
            )
    print('Save OK')
