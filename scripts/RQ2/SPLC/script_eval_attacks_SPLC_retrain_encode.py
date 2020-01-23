import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

import copy
import pickle

import sys


header = np.genfromtxt('../../../data/MOTIV/train_data_after_preprocessing.csv',dtype=str,delimiter=',')
header = header[0]

## load previously trained ML model (SVM linear)
filename = "../../../data/MOTIV/model_classif_after_preprocessing.txt"
#filename = "../../../data/MOTIV/augmented_model_classif_after_preprocessing.txt"
clf = joblib.load(filename)
clf.max_iter=1000000000

### load data and labels to clone and create adversarial example
data_full = np.loadtxt('../../../data/MOTIV/test_data_after_preprocessing.csv',delimiter=',',skiprows=1)
### load training set to prepare retraining
data_full_train = np.loadtxt('../../../data/MOTIV/train_data_after_preprocessing.csv',delimiter=',',skiprows=1)

data=data_full[:,0:-1]
value = data_full[:,-1]

#check for initial performances
yc = clf.predict(data)
acc_score = accuracy_score(value, yc)
print(acc_score)

data_train=data_full_train[:,0:-1]
label_train = data_full_train[:,-1]


max_iter = 10
nb_init_pts=25
max_disp=20
step = np.logspace(-4,2,num=7,endpoint=True)

##number of feature for data
nb_f_total = (data_full[1].size)-1

##class from which attack starts
label = 1

##absolute importance feature
g = clf.coef_
g_norm = g/np.linalg.norm(g)
abs_g = np.absolute(g_norm)

abs_g_ord = sorted(abs_g[0], key=float, reverse=True)

## repeat for each step size of displacement
for stp in step:

    ##repeat to minimize the impact of the random choice of attack points
    #for nb_iter in range(0,max_iter):
    nb_iter=0
    while nb_iter < max_iter:

        nb_attack_pt=0

        #reset all the data sets
        data=data_full[:,0:-1]
        value = data_full[:,-1]

        data_train=data_full_train[:,0:-1]
        label_train = data_full_train[:,-1]

        while nb_attack_pt < nb_init_pts:
            ##retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
            idx_candidates = np.where(value == label)

            ##choose one point
            #select nb_init_pts points randomly in candidates and make them move
            idx_c = np.random.choice(idx_candidates[0].size, nb_init_pts)
            #print idx_c
            pts_to_move = copy.copy(data[(idx_candidates[0][idx_c]),:])

            for pt in pts_to_move:
                ## predict the class of point used to attack (should be the same as value)
                pred = clf.predict(pt.reshape(1,-1))
        
                ##successive move for the same point
                for nb_step in range(0,max_disp):

                    ##for each feature
                    cpt_nb_feat=0
                    i=0
                    while cpt_nb_feat < nb_f_total:

                        ##select the most important feature not treated yet
                        idx = np.where(abs_g == abs_g_ord[i])
                        nb_feat = idx[1]


                        ##in case multiple features have the same importance
                        for single_feat in nb_feat:

                            ##move in the direction of the gradient for current feature
                            if label == 0:
                                pt[single_feat] += stp*g_norm[0][single_feat]
                            else:
                                pt[single_feat] -= stp*g_norm[0][single_feat]          


                            ## apply type constraints to the values if any (Boolean, between 0 and 1, etc.)
                            #print single_feat
                            if single_feat <= 119 :
                                pt[single_feat] = round(pt[single_feat])
                            if single_feat > 119 and single_feat <= 129 :
                                if pt[single_feat] < 0:
                                    pt[single_feat] = 0
                                if pt[single_feat] > 1:
                                    pt[single_feat] = 1
                            if single_feat == 130:
                                pt[single_feat] = 0
                            if single_feat > 130 and single_feat <= 144:
                                if pt[single_feat] < 0:
                                    pt[single_feat] = 0
                                if pt[single_feat] > 1:
                                    pt[single_feat] = 1

                        cpt_nb_feat = cpt_nb_feat+1
                        i=i+1

                nb_attack_pt += 1

                #add adversarial data (last position)
                pt = np.reshape(pt,(1,-1))
                #data = np.append(data,pt,axis=0)
                data_train = np.append(data_train,pt,axis=0)
                label_train = np.append(label_train,label)

        df = pd.DataFrame(data=np.c_[data_train,label_train])
        df.to_csv("../../../results/MOTIV/RQ2/SPLC/original/train_data_for_retraining_25_attack_pts_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_iter_"+str(nb_iter)+".csv",header=header,index=False)
        ###RETRAINING

        #parameter tuning?
        #not done yet
        
        #retraining
        clf.fit(data_train,label_train)
        yc = clf.predict(data)


        acc_score = accuracy_score(value, yc)

        f=open("../../../results/MOTIV/RQ2/SPLC/original/perf_predict_after_retraining_25_attack_pts_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_iter_"+str(nb_iter)+".txt","w")
        orig_stdout = sys.stdout
        sys.stdout = f

        print (stp)
        print(max_disp)
        print (acc_score)

        sys.stdout = orig_stdout
        f.close()

        nb_iter = nb_iter+1


