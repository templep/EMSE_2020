import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

import copy
import pickle

import sys



## load previously trained ML model (SVM linear)
filename = "../../../data/MOTIV/model_classif_after_preprocessing.txt"
#filename = "../../../data/MOTIV/augmented_model_classif_after_preprocessing.txt"
clf = joblib.load(filename)
clf.max_iter=1000000000


### load data and labels to clone and create adversarial example
data_full = np.loadtxt('../../../data/MOTIV/test_data_after_preprocessing.csv',delimiter=',',skiprows=1)
#data_full = np.loadtxt('../../../data/MOTIV/augmented_test_data_after_preprocessing.csv',delimiter=',',skiprows=1)


nb_init_pts=1
test_set_size=4000

## step to advance
## from 10^-10 to 100
step = np.logspace(-6,6,num=7,endpoint=True)
#step = np.logspace(-3,2,num=6,endpoint=True)
#print step


##number of feature for data
nb_f_total = (data_full[1].size)-1
#print "nb feat:"+str(nb_f_total)

##class from which attack starts
label = 0
#label = 1

##absolute importance feature
g = clf.coef_
g_norm = g/np.linalg.norm(g)
abs_g = np.absolute(g_norm)

abs_g_ord = sorted(abs_g[0], key=float, reverse=True)

max_iter = 10

#max_disp=20
#max_disp=50
max_disp=100

## repeat for each step size of displacement
for stp in step:

    ## init with zeros the output array containing the number of successful attacks per run
    nb_succeeded_attack = [0 for row in range(0,max_iter)]

    ##repeat to minimize the impact of the random choice of attack points
    #for nb_iter in range(0,max_iter):
    nb_iter=0
    while nb_iter < max_iter:

        data=data_full[:,0:-1]
        value = data_full[:,-1]

        nb_attack_pt=0

        ## generate 4k attack points
        while nb_attack_pt < test_set_size:

            ##retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
            idx_candidates = np.where(value == label)

            #select nb_init_pts points randomly in candidates and make them move
            idx_c = np.random.choice(idx_candidates[0].size, nb_init_pts)
            #print idx_c
            pts_to_move = copy.copy(data[(idx_candidates[0][idx_c]),:])

            ##choose one point
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
                            #if single_feat == 22:
                            #    if pt[single_feat] < 0:
                            #        pt[single_feat] = 0
                            #    if pt[single_feat] > 1:
                            #        pt[single_feat] = 1
                            #    pt[single_feat] = round(pt[single_feat])	
                            ##between index 23 and 28, do not change values (unbounded float) but should not be over 256 and below 0 in theory (for colors)
                            #if single_feat >= 29 and single_feat <= 34:
                            #    if pt[single_feat] < 0:
                            #        pt[single_feat] = 0
                            #    if pt[single_feat] > 1:
                            #        pt[single_feat] = 1
                            ## end constraints

                        cpt_nb_feat = cpt_nb_feat+1
                        i=i+1

                nb_attack_pt += 1
                #print(str(nb_iter) +";"+ str(nb_attack_pt))

                ##after displacement, predict again to check if any changes
                yc= clf.predict(pt.reshape(1,-1))

                ##if prediction is the same as before... Attack failed
                if yc != label:
                    #print "Attack succeeded"
                    ##increase number of failed attacks
                    nb_succeeded_attack[nb_iter] += 1

                ##put attack pt in the data set to be able to pick it for a new attack
                ##added point are removed when launching a new evaluation iteration
                pt = np.reshape(pt,(1,-1))
                data = np.append(data,pt,axis=0)
                value = np.append(value,label)

        if label == 0:
            np.savetxt("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/non_acc/data_test_training_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.csv",data,delimiter=',')
            #np.savetxt("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/non_acc/balanced_data_test_training_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.csv",data,delimiter=',')
        else:
            np.savetxt("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/acc/data_test_training_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".csv",data,delimiter=',')
            #np.savetxt("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/acc/balanced_data_test_training_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".csv",data,delimiter=',')
        nb_iter +=1

    if label == 0:
        out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.txt"
        #out_filename="balanced_perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.txt"
        f=open("../../../results/MOTIV/RQ1.1/SPLC/perf_predict_adv_attack_nonacc_norm/"+out_filename,"w")
    else:
        out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".txt"
        #out_filename="balanced_perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".txt"
        f=open("../../../results/MOTIV/RQ1.1/SPLC/perf_predict_adv_attack_norm/"+out_filename,"w")
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

