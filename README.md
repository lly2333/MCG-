# MCG+：
This project is the for the paper: MCGNet+: An improved motor imagery classification based on Cosine similarity
# Prerequistes
* Python 3.6, tensorflow-1.2.0
* the code is built upon the tensorflow framework
* pip install -r requirts.txt
# How to run
##dataset
* the dataset is come from the BNCI-competition2014, which can be load in:http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014001.html
* the dataset need to be preprocessed, we didn't use the raw data, instead we use the DE features, and the code for preprocessing the dataset and extracting the DE features can be found in the directory : process/process_data.py
* we use the mutual information to generate the adjency matrix, this part can be find in: process/generate_feature.py
##the configuration
the  settings of the model can be found in: config/run.config
##loss setting
In our paper, we use a loss refine method, and in the tensorflow framework the loss should be loss = abs(loss-0.8), which can be achieved by changing the: tf/Lib/site-packages/keras/engine/training.py
##run
run python train.py
