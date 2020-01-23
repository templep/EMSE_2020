import numpy as np
from sklearn.externals import joblib

import copy
import pickle

import sys


#load data
input_path="../../../data/JHipster/data_clean/"
input_filename="jhipster_transformed_ready_gen.csv"
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
#del X[nb_col]
#print(X.shape)
#print(X[:,nb_col-2])
#print(Y.shape)

nb_config=200

from sklearn import mixture as mxt

gmm = mxt.BayesianGaussianMixture(n_components=2)
gmm.fit(X,Y)
#print(gmm.get_params())
x_smp,y_smp = gmm.sample(n_samples=nb_config)

#count number of occurence of each class in the sample
from collections import Counter
print(Counter(y_smp))


#test does sampling retrieve config from X?
#cpt=0
#for x in x_smp:
#	if(x not in X):
#		print("Nope")
#		cpt = cpt+1

#print(cpt)

#EM might create float values configurations while we only used integers (Boolean)
x_conv = x_smp.astype(int)
#in case values are closer to -1 than 0 before casting to int
#np.where(x_conv == -1,0,x_conv)
x_conv[x_conv < 0] = 0
y_conv = y_smp.astype(int)

#convert initial data to int
X = X.astype(int)
Y = Y.astype(int)

#retrieve closest configuration from the original set with some kind of edition distance
#for i1,x in enumerate(x_conv):
#	sum_x = np.sum(x)
# 	dst = 0
#	dst_min=  np.iinfo(int).max
#	index_min_dst=0
#	for i2,x2 in enumerate(X):
#		sum_x2 = np.sum(x2)
#		dst = np.absolute(sum_x-sum_x2)
#		if(dst < dst_min):
#			index_min_dst=i2
#			print("index min distance:"+str(index_min_dst))
#			dst_min = dst
#	np.copyto(x_conv[i1,:],X[index_min_dst,:])
#	y_conv[i1] = Y[index_min_dst]

cpt=0
for x in x_conv:
	if(x not in X):
		print("Nope")
		cpt = cpt+1

print(cpt)
#expected value of cpt after conversion is 0

#ensure that only different configurations are in the sampled set
nb_elem_resamp=0
x_conv_rsmp=x_conv
y_conv_rsmp=y_conv
for x in x_conv:
	idx = np.where(x_conv in x)
	idx = list(idx)
	print("Size list: "+str(len(idx)))
	print(idx)
	while len(idx) > 1:
		np.delete(x_conv_rsmp,idx[1],0)
		np.delete(y_conv_rsmp,idx[1],0)
		del(idx[1])
		nb_elem_resamp = nb_elem_resamp+1
print("NB element to resample: "+str(nb_elem_resamp))

while nb_elem_resamp != 0:
	nb_elem_resamp=0
	x_resmp,y_resmp = gmm.sample(n_samples=nb_elem_resamp)
	x_resampled_conv = x_resmp.astype(int)
	y_resampled_conv = y_resmp.astype(int)
	for x,y in x_resampled_conv,y_resampled_conv:
		if(x not in x_conv_rsmp):
			x_conv_rsmp.append(x)
			y_conv_rsmp.append(y)
		else:
			nb_elem_resamp=nb_elem_resamp+1


print(len(x_conv_rsmp))
print(Counter(y_conv_rsmp))



output_filename="data_resampled_"+str(nb_config)+"_configurations.csv"
import pandas as pd 
data = pd.concat([pd.DataFrame(x_conv_rsmp),pd.DataFrame(y_conv_rsmp)],axis=1,ignore_index=True)
data.to_csv(input_path+output_filename,index=False)


