import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

import copy
import pickle

import sys

import random


#load data
input_path="../../../data/JHipster/data_clean/"
input_filename="jhipster_transformed_ready_build.csv"
input_filename_smp="data_randomly_sampled_200_configurations.csv"
input_file=input_path+input_filename
input_file_smp=input_path+input_filename_smp

data_full = np.loadtxt(input_file, delimiter=',',skiprows=1)
data_smp_train = np.loadtxt(input_file_smp, delimiter=',',skiprows=1)

#test nb lines and columns
nb_line = len(data_full)
nb_col = data_full[1].size

#learn a classifier
clf=SVC(kernel='linear')
clf.max_iter=1000000000

##separated class from the rests of the features
Y=data_full[:,nb_col-1]
X = np.delete(data_full,nb_col-1,axis=1)
Y_smp = data_smp_train[:,nb_col-1]
X_smp = np.delete(data_smp_train,nb_col-1,axis=1)

for x in X_smp:
    idx = np.where((X == x.T).all(axis=1))
    X = np.delete(X,idx,axis=0)
    Y = np.delete(Y,idx,axis=0)

#remove configurations from the total data set that are in the training set;
#is not necessarily the difference of lengths because there might exist configurations appearing twice or more
import sys
if(len(X) > nb_line-len(X_smp)):
    print("Expected number of lines:"+str(nb_line-len(X_smp)))
    print("Retrieved number of lines:"+str(len(X)))
    sys.exit("Expected number of line in test set is not matched")

clf.fit(X_smp,Y_smp)



##class from which attack starts
label = 1
#label = 0

nb_init_pts=1
test_set_size=1000

max_iter = 10
max_disp=20
#max_disp=50
#max_disp=100

## step to advance
step = np.logspace(-6,6,num=7,endpoint=True)
#step = np.logspace(-3,2,num=6,endpoint=True)
#print step

##absolute importance feature
g = clf.coef_
g_norm = g/np.linalg.norm(g)
abs_g = np.absolute(g_norm)

abs_g_ord = sorted(abs_g[0], key=float, reverse=True)

data=np.empty((test_set_size,nb_col-1))
value=np.empty((test_set_size,1))

min_grad = min(min(g_norm))
max_grad = max(max(g_norm))

print("Entering attack loop")
## repeat for each step size of displacement
for stp in step:

    ## init with zeros the output array containing the number of successful attacks per run
    nb_succeeded_attack = [0 for row in range(0,max_iter)]

    ##repeat to minimize the impact of the random choice of attack points
    #for nb_iter in range(0,max_iter):
    nb_iter=0
    while nb_iter < max_iter:

        nb_attack_pt=0

        ## generate 4k attack points
        while nb_attack_pt < test_set_size:

            ##retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
            idx_candidates = np.where(Y == label)

            #select nb_init_pts points randomly in candidates and make them move
            idx_c = np.random.choice(idx_candidates[0].size, nb_init_pts)
            pts_to_move = copy.copy(X[(idx_candidates[0][idx_c]),:])
            ##choose one point
            nb_proc=0
            for pt in pts_to_move:
        
                ## predict the class of point used to attack (should be the same as value)
                pred = clf.predict(pt.reshape(1,-1))
        
                ##successive move for the same point
                for nb_step in range(max_disp):

                    ##for each feature
                    cpt_nb_feat=0
                    i=0
                    while cpt_nb_feat < nb_col-1:

                        ##select the most important feature not treated yet
                        idx = np.where(abs_g == abs_g_ord[i])
                        nb_feat = idx[1]


                        ##in case multiple features have the same importance
                        for single_feat in nb_feat:

                            ##move in the direction of the gradient for current feature
                            #pt[single_feat] -= stp*np.random.rand(1)

                            change_feat = random.choice([True, False])
                            if change_feat:
                                rand_dir = random.choice([-1, +1])
                                rand_grad = random.uniform(min_grad,max_grad)
                                pt[single_feat] += rand_dir*stp*rand_grad

                        cpt_nb_feat = cpt_nb_feat+1
                        i=i+1

                nb_attack_pt += 1

                for f in range(nb_col-1):
                    if pt[f] > 1:
                        pt[f] = 1
                    if pt[f] < 0:
                        pt[f] = 0

                ##after displacement, predict again to check if any changes
                yc= clf.predict(pt.reshape(1,-1))

                ##if prediction is the same as before... Attack failed
                if yc != label:
                    nb_succeeded_attack[nb_iter] += 1

                ##put attack pt in the data set to be able to pick it for a new attack
                ##added point are removed when launching a new evaluation iteration
                pt = np.reshape(pt,(1,-1))
                data.put(nb_proc,pt)
                value.put(nb_proc,yc)
                nb_proc=nb_proc+1

        #check and force the validity wrt the FM
        for x in data:
                if x[2] == 1:
                        if x[6] == 0 and x[5] == 0:
                                x[6] = 1
                                #could be x[5] = 1 also
                        if x[10] == 1:
                                x[10] = 0
                if x[0] == 1 or x[1] == 1:
                        if x[7] == 0 and x[10] == 0:
                                x[7] = 1
                                #could be x[10] = 1 also
                if x[1] == 0 and (x[3] == 0 or x[0] == 0):
                        x[20] = 0
                if x[8] == 1:
                        if x[19] == 0 and x[21] == 0:
                                x[21] = 1
                                #could be x[19] = 1 also
                if x[21] == 1:
                        x[36] = 0
                if x[21] == 1 and x[32] == 1:
                        if x[23] == 0 and x[24] == 0 and x[25] == 0:
                                x[23] =1
                                #could be x[24] = 1 or x[25] =1 also
                if x[21] == 1 and x[34] == 1:
                        if x[23] == 0 and x[24] == 0 and x[30] == 0:
                                x[23] =1
                                #could be x[24] = 1 or x[30] =1 also
                if x[21] == 1 and x[35] == 1:
                        if x[23] == 0 and x[24] == 0 and x[27] == 0:
                                x[23] =1
                                #could be x[24] = 1 or x[27] =1 also
                if x[21] == 1 and x[37] == 1:
                        if x[23] == 0 and x[24] == 0 and x[28] == 0:
                                x[23] =1
                                #could be x[24] = 1 or x[28] =1 also
                if x[5] == 1 and x[10] == 0:
                        x[7] = 1
                if x[18] == 1 or x[19] == 1 or x[20] == 1:
                        x[29] = 1
                        x[36] = 1
                if x[3] == 1:
                        x[10] = 1
                if (x[19] == 1 or x[18] == 1) and (x[0] == 0 or x[20] == 1):
                        x[14] = 1
                if x[0] == 1:
                        x[12] = 1
                if x[18] == 1 or x[2] == 0 or (x[9] == 0 and x[7] == 0):
                        x[40] = 0
                if x[21] == 0:
                        x[39] = 0
                if (x[2] == 0 and x[0] == 0) or (x[20] == 0 and x[12] == 0):
                        x[15] = 0
                if (x[2] == 0 and x[0] == 0):
                        x[16] = 0
                if (x[1] == 1 or x[3] == 1):
                        x[44] = 1
                        x[45] = 0
                if x[0] == 1 or x[2] == 1:
                        x[45] = 1
                        x[42] = 1
                        #could be x[43] = 1 also


        if label == 0:
            np.savetxt("../../../results/JHipster/RQ1.3/SPLC/non_acc/test_adv_attack_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_0.csv",data.round().astype(int),delimiter=',')
        else:
            np.savetxt("../../../results/JHipster/RQ1.3/SPLC/acc/test_adv_attack_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_1.csv",np.array(data).round().astype(int),delimiter=',')
        nb_iter +=1

    if label == 0:
        #out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.txt"
        out_filename="perf_prediction_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.txt"
        f=open("../../../results/JHipster/RQ1.3/SPLC/check_valid/"+out_filename,"w")
    else:
        #out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".txt"
        out_filename="perf_prediction_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".txt"
        f=open("../../../results/JHipster/RQ1.3/SPLC/check_valid/"+out_filename,"w")
    orig_stdout = sys.stdout
    sys.stdout = f

    #print str(nb_succeeded_attack)
    print (stp)
    print(max_disp)
    print (nb_succeeded_attack)
    print ("mean: "+ str(np.mean(nb_succeeded_attack,axis=0)))
    #print "avg: "+str(np.average(nb_succeeded_attack,axis=1))
    print ("std_dev: "+str(np.std(nb_succeeded_attack,axis=0)))
    #print "var: "+str(np.var(nb_succeeded_attack,axis=1))

    print ("max: "+str(np.amax(nb_succeeded_attack,axis=0)))
    print ("min: "+str(np.amin(nb_succeeded_attack,axis=0)))
    #print pt
    sys.stdout = orig_stdout
    f.close()

