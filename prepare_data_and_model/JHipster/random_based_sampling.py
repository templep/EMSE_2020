import numpy as np
from sklearn.externals import joblib

import copy
import pickle

import sys
from collections import Counter


#load data
input_path="../../../data/JHipster/data_clean/"
input_filename="jhipster_transformed_ready_build.csv"
input_file=input_path+input_filename

data_full = np.loadtxt(input_file, delimiter=',',skiprows=1)

#test nb lines and columns
nb_line = len(data_full)
#print(nb_line)
nb_col = data_full[1].size
#print("shape of the matrix: "+str(data_full.shape))

##separated class from the rests of the features
Y=data_full[:,nb_col-1]
X = np.delete(data_full,nb_col-1,axis=1)

Y.astype(int)
X.astype(int)

import copy
nb_config=200
###choose configurations for class 0
##retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
idx_candidates = np.where(Y == 0.0)

#select nb_init_pts points randomly in candidates and make them move

idx_c = np.random.choice(idx_candidates[0], size=round(nb_config/2),replace=False)
chosen_config = copy.copy(X[idx_c,:])
y_smp = copy.copy(Y[idx_c])


###choose configurations for class 1 and append to first pts
##retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
idx_candidates = np.where(Y == 1)


#select nb_init_pts points randomly in candidates and make them move
idx_c = np.random.choice(idx_candidates[0], size=round(nb_config/2),replace=False)
temp = copy.copy(X[idx_c,:])
chosen_config = np.append(chosen_config,temp,axis=0)
temp = copy.copy(Y[idx_c])
y_smp = np.append(y_smp,temp,axis=0)


#read and loaded as float, convert into int
x_conv = chosen_config.astype(int)
y_conv = y_smp.astype(int)

for x in x_conv:
	idx_config_same = np.where((x_conv == x.T).all(axis=1))
	if(len(idx_config_same) != 1):
		print(idx_config_same)

output_filename="data_randomly_sampled_"+str(nb_config)+"_configurations.csv"
import pandas as pd 
data = pd.concat([pd.DataFrame(x_conv),pd.DataFrame(y_conv)],axis=1,ignore_index=True)
data.to_csv(input_path+output_filename,index=False)


