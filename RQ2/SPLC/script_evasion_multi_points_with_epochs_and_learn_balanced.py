import numpy as np
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import copy
import subprocess
import fileinput
import pandas as pd

from sklearn.externals import joblib

############################################
# we want to generate points of class '0' that originally come from class '1'
# we move points from class '0' towards class '1' thanks to gradient descent techniques
#
# this is a new version (of the script_evasion.py) in which we want generate more points automatically
# points are chosen randomly in the class '0'
# a mini-batch of adversarial points are computed at each step
# the model is updated
#
################# ALGO ##############
#compute gradient
#choose one point in the test set and go to the min of gradient
#retrieve config
#check at some point ?

############# description of dims
# 15 + 16*4 + 8*5 bool features (between 0 and 1)
# 12 float feautres (between 0 and 1)
# 2 bool features (0 or 1)
# 6 float features (between 0 and 255)
# 6 float features (between 0 and 1)
# total nb_features = 119 + 12 + 2 + 6 + 6 = 145
##############################


## load previously trained ML model (SVM linear)
filename = "../../../data/MOTIV/model_classif_balanced.txt"
clf = joblib.load(filename)
clf.max_iter=1000000000

#absolute importance feature
g = clf.coef_
g_norm = g/np.linalg.norm(g)
abs_g = np.absolute(g_norm)
abs_g_ord = sorted(abs_g[0], key=float, reverse=True)

### load data and labels to clone and create adversarial example
data_full = np.loadtxt('../../../data/MOTIV/augmented_train_data_format_weka.csv',delimiter=',',skiprows=1)
#data_full = np.loadtxt('../data/train_format_weka_with_few_non-accept.csv',delimiter=',',skiprows=1)
header = np.genfromtxt('../../../data/MOTIV/augmented_train_data_format_weka.csv',dtype=str,delimiter=',')
header = header[0]

#data=data_full[:,0:-1]
#value = np.around(data_full[:,-1])


##load data test
data_full_test = np.loadtxt('../../../data/MOTIV/test_format_weka.csv',delimiter=',',skiprows=1)
#data_full_test = np.loadtxt('../data/augmented_test_data_format_weka',delimiter=',',skiprows=1)
data_test=data_full_test[:,0:-1]

value_test = data_full_test[:,-1]


##call weka J48 implem
##evaluate J48 on initial datasets (train on train_format_weka.arff and test on test_format_weka.arff)
#subprocess.call(["java -cp ~/weka/weka-3-8-1/weka.jar weka.classifiers.trees.J48 -t ../data/augmented_train_data_format_weka.arff -T ../data/test_format_weka.arff -C 0.25 -M 2 -d j48.model -o > ../changing_params/expe_final_retrain_python/eval_stat_balanced.txt"],shell=True)
subprocess.call(["java -cp ../../../weka/weka.jar weka.classifiers.trees.J48 -t ../../../data/MOTIV/augmented_train_data_format_weka.arff -T ../../../data/MOTIV/augmented_test_data_format_weka.arff -C 0.25 -M 2 -d j48.model -o > ../../../results/MOTIV/retrain/eval_stat_balanced_augmented.txt"],shell=True)

##save tree in order to extract constraints
#subprocess.call(["java -cp ~/weka/weka-3-8-1/weka.jar weka.classifiers.trees.J48 -t ../data/augmented_train_data_format_weka.arff -T ../data/test_format_weka.arff -C 0.25 -M 2 -d j48.model -g > ../changing_params/expe_final_retrain_python/init_tree_balanced.txt"],shell=True)
subprocess.call(["java -cp ../../../weka/weka.jar weka.classifiers.trees.J48 -t ../../../data/MOTIV/augmented_train_data_format_weka.arff -T ../../../data/MOTIV/augmented_test_data_format_weka.arff -C 0.25 -M 2 -d j48.model -g > ../../../results/MOTIV/retrain/init_tree_balanced_augmented.txt"],shell=True)

#nb_epochs=100
#nb_init_pts=10
nb_init_pts=25
#nb_init_pts=1

max_disp=20

## step to advance
#step = 0.0001
step = np.logspace(-4,2,num=7,endpoint=True)

##number of feature for data
nb_f_total = (data_full[1].size)-1
#print "nb feat:"+str(nb_f_total)

##class from which attack starts
label = 1


## repeat for each step size of displacement
for stp in step:

    #a full iteration is when all nb_init_pts have been moved nb_epochs times for each feature and that we start from the beginning with the initial set of data (no adv configurations)
    for it in range(0,10):

        data=data_full[:,0:-1]
        value = np.around(data_full[:,-1])
        clf.fit(data,value)

        #one epoch is a complete round of attack (all points have been moved over each feature and a certain number of time)
        for nb_epochs in range(0,1):

            #retrieve candidates (configs which are non-acceptable -> i.e., with value = 1)
            idx_candidates = np.where(value == label)

            nb_pt=0

            ## generate nb_init_pts adv config
            while nb_pt < nb_init_pts:

                #select nb_init_pts points randomly in candidates and make them move
                idx_c = np.random.choice(idx_candidates[0].size, 1)
                pts_to_move = copy.copy(data[(idx_candidates[0][idx_c]),:])
                pt_to_move=pts_to_move[0]
                
                #predict the class of point used to attack (should be the same as value)
                pred = clf.predict(pt_to_move.reshape(1,-1))
        
                #successive move for the same point
                for nb_iter in range(0,max_disp):

                    #for each feature
                    cpt_nb_feat=0
                    i=0
                    while cpt_nb_feat < nb_f_total:

                        #select the most important feature not treated yet
                        idx = np.where(abs_g == abs_g_ord[i])
                        nb_feat = idx[1]


                        #in case multiple features have the same importance
                        for single_feat in nb_feat:

                            #move in the direction of the gradient for current feature
                            if label == 0:
                                pt_to_move[single_feat] += stp*g_norm[0][single_feat]
                            else:
                                pt_to_move[single_feat] -= stp*g_norm[0][single_feat] 

                            ## apply type constraints to the values if any (Boolean, between 0 and 1, etc.)
                            #print single_feat
                            if single_feat <= 9 :
                                pt_to_move[single_feat] = round(pt_to_move[single_feat])
                            if single_feat > 9 and single_feat <= 19 :
                                if pt_to_move[single_feat] < 0:
                                    pt_to_move[single_feat] = 0
                                if pt_to_move[single_feat] > 1:
                                    pt_to_move[single_feat] = 1
                            if single_feat == 20:
                                pt_to_move[single_feat] = 0
                            if single_feat == 21:
                                if pt_to_move[single_feat] < 0:
                                    pt_to_move[single_feat] = 0
                                if pt_to_move[single_feat] > 1:
                                    pt_to_move[single_feat] = 1
                            if single_feat == 22:
                                if pt_to_move[single_feat] < 0:
                                    pt_to_move[single_feat] = 0
                                if pt_to_move[single_feat] > 1:
                                    pt_to_move[single_feat] = 1
                                pt_to_move[single_feat] = round(pt_to_move[single_feat])
                            #between index 24 and 29, do not change values (unbounded float) but should not be over 256 and below 0 in theory (for colors)
                            if single_feat >= 29 and single_feat <= 34:
                                if pt_to_move[single_feat] < 0:
                                    pt_to_move[single_feat] = 0
                                if pt_to_move[single_feat] > 1:
                                    pt_to_move[single_feat] = 1
                            ## end constraints

                        cpt_nb_feat = cpt_nb_feat+1
                        i=i+1
                     
                #final position of the attack   
                #nb attack performed
                nb_pt=nb_pt+1

                #after displacement, predict again to check if any changes
                yc= clf.predict(pt_to_move.reshape(1,-1))

                #if nb_iter % 10 == 0:
                #np.savetxt("./samples/version_epochs_n_pts/config_adv_"+str(nb_epochs)+"_"+str(nb_pt)+"_"+str(nb_iter)+".txt",pt)

                #add adversarial data (last position)
                pt_to_move = np.reshape(pt_to_move,(1,-1))
                data = np.append(data,pt_to_move,axis=0)
                value = np.append(value,label)

        print("end attack epochs: " + str(nb_epochs))

        #learn new model with adversarial data
        clf.fit(data,value)
        #filename = "../changing_params/expe_final_retrain_python/model_classif_after_retrain_epoch_"+str(nb_epochs)+"_with_step_"+str(stp)+"_final_norm_balanced.txt"
        filename = "../../../results/MOTIV/retrain/model_classif_after_retrain_epoch_"+str(nb_epochs)+"_with_step_"+str(stp)+"_final_norm_balanced_augmented.txt"
        joblib.dump(clf,filename)
        print("end retrain")

        #concatenate all config to classes
        #data_full_after_attack = np.c_[data,value]
        #data_full_header = np.append(header,data_full_after_attack)
        #save original config + adv configs to new file
        #np.savetxt("./txt/data_after_attack_1_epochs_50_iter_25_pts_001_eps.txt",data_full_after_attack)

        df = pd.DataFrame(data=np.c_[data,value])
        #df.to_csv("../changing_params/expe_final_retrain_python//csv/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.csv",header=header,index=False)
        df.to_csv("../../../results/MOTIV/retrain/csv/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced_augmented.csv",header=header,index=False)

        print("end dumping")

        ####RETRAIN

        print("begin retrain part")
        #subprocess.call(["java -cp ~/weka/weka-3-8-1/weka.jar weka.core.converters.CSVLoader ../changing_params/expe_final_retrain_python/csv/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.csv > ../changing_params/expe_final_retrain_python/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff"],shell=True)
        subprocess.call(["java -cp ../../../weka/weka.jar weka.core.converters.CSVLoader ../../../results/MOTIV/retrain/csv/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced_augmented.csv > ../../../results/MOTIV/retrain/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced_augmented.arff"],shell=True)
        print("end convert csv to arff")

        #fi = fileinput.input("../changing_params/expe_final_retrain_python/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff",inplace=True)
        fi = fileinput.input("../../../results/MOTIV/retrain/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced_augmented.arff",inplace=True)
        for line in fi:
            if line.startswith("@attribute label_(0_if_usable_and_1_if_not) numeric"):
                line = "@attribute class {0,1}"
            print(line)
        fi.close()

        print("begin eval")
        #subprocess.call(["java -cp ~/weka/weka-3-8-1/weka.jar weka.classifiers.trees.J48 -t ../changing_params/expe_final_retrain_python/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff -T ../data/test_format_weka.arff -C 0.25 -M 2 -d j48.model -o > ../changing_params/expe_final_retrain_python/eval_stat_after_attack_1_epochs_"+str(nb_init_pts)+"_pts_"+str(it+1)+"_disp_"+str(max_disp)+"_stp_"+str(stp)+"_norm_balanced.txt"],shell=True,timeout=900)
        subprocess.call(["java -cp ../../../weka/weka.jar weka.classifiers.trees.J48 -t ../../../results/MOTIV/retrain/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff -T ../../../data/MOTIV/augmented_test_data_format_weka.arff -C 0.25 -M 2 -d j48.model -o > ../../../results/MOTIV/retrain/eval_stat_after_attack_1_epochs_"+str(nb_init_pts)+"_pts_"+str(it+1)+"_disp_"+str(max_disp)+"_stp_"+str(stp)+"_norm_balanced_augmented.txt"],shell=True,timeout=900)

        print("begin tree")
        ##save tree in order to extract constraints
        #subprocess.call(["java -cp ~/weka/weka-3-8-1/weka.jar weka.classifiers.trees.J48 -t ../changing_params/expe_final_retrain_python/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff -T ../data/test_format_weka.arff -C 0.25 -M 2 -d j48.model -g > ../changing_params/expe_final_retrain_python/tree_after_attack_"+str(nb_init_pts)+"_pts_"+str(it+1)+"_disp_"+str(max_disp)+"_stp_"+str(stp)+"_norm_balanced.txt"],shell=True,timeout=900)
        subprocess.call(["java -cp ../../../weka/weka.jar weka.classifiers.trees.J48 -t ../../../results/MOTIV/retrain/arff/data_after_attack_1_epochs_"+str(max_disp)+"_iter_"+str(nb_init_pts)+"_pts_"+str(stp)+"_eps_"+str(it+1)+"_iter_norm_balanced.arff -T ../../../data/MOTIV/augmented_test_data_format_weka.arff -C 0.25 -M 2 -d j48.model -g > ../../../results/MOTIV/retrain/tree_after_attack_"+str(nb_init_pts)+"_pts_"+str(it+1)+"_disp_"+str(max_disp)+"_stp_"+str(stp)+"_norm_balanced_augmented.txt"],shell=True,timeout=900)

        print("end iter "+str(it))


