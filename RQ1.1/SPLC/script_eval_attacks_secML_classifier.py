import pandas as pd
import numpy as np
import sys

from secml.data import CDataset

### load data and labels to clone and create adversarial example
#data_train = np.loadtxt('../../../data/MOTIV/train_data_after_preprocessing.csv',delimiter=',',skiprows=1)
data_train = np.loadtxt('../../../data/MOTIV/augmented_train_data_after_preprocessing.csv',delimiter=',',skiprows=1)
#data_test = np.loadtxt('../../../data/MOTIV/test_data_after_preprocessing.csv',delimiter=',',skiprows=1)
data_test = np.loadtxt('../../../data/MOTIV/augmented_test_data_after_preprocessing.csv',delimiter=',',skiprows=1)

n_tr = len(data_train)
n_ts=len(data_test)
#all rows in the data_train and data_test have the same number of columns
nb_col = data_train[1].size

##separated class from the rests of the features => last column represents the class
Y_tr=data_train[:,nb_col-1]
Y_tr.astype(int)
data_train=np.delete(data_train,nb_col-1,axis=1)

Y_test = data_test[:,nb_col-1]
Y_test.astype(int)
data_test=np.delete(data_test,nb_col-1,axis=1)

tr_set = CDataset(data_train,Y_tr)
ts_set = CDataset(data_test,Y_test)

# create a splitter for training data -> training and validation sets
from secml.data.splitter import CDataSplitterKFold
#default splitter for cross-val -> num_folds=3 and random_state=None
xval_splitter = CDataSplitterKFold()

##train a classifier (SVM with linear kernel)
from secml.ml.classifiers import CClassifierSVM
clf_lin = CClassifierSVM()

#problem seems linearly decidable -> try a logistic regression classifier without any parameter estimations
from secml.ml.classifiers import CClassifierLogistic
clf_l= CClassifierLogistic()

xval_lin_params = {'C': [0.01, 0.1, 1, 10, 100]}

# Select and set the best training parameters for the linear classifier
print("Estimating the best training parameters for linear kernel...")
best_lin_params = clf_lin.estimate_parameters(
    dataset=tr_set,
    parameters=xval_lin_params,
    splitter=xval_splitter,
    metric='accuracy',
    perf_evaluator='xval'
)

clf_lin.fit(tr_set)

# Select and set the best training parameters for the linear classifier
#print("Estimating the best training parameters for linear kernel...")
#best_lin_params = clf_l.estimate_parameters(
#    dataset=tr_set,
#    parameters=xval_lin_params,
#    splitter=xval_splitter,
#    metric='accuracy',
#    perf_evaluator='xval'
#)

#clf_l.fit(tr_set)

import random
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.attacks.evasion import CAttackEvasionPGD
#perform adversarial attacks
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
dmax = 10.0 # Maximum perturbation    values to test: 10^-1; 5*10^-1; 1; 5; 10
lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded

solver_params = {'eta' : 0.01}

#set lower bound and upper bound respectively to 0 and 1 since all features are Boolean
pgd_attack = CAttackEvasionPGD(
    classifier=clf_lin,
    surrogate_classifier=clf_lin,
    surrogate_data=tr_set,
    distance=noise_type,
    dmax=dmax,
    lb=lb, ub=ub,
    solver_params=solver_params)

#pgd_attack2 = CAttackEvasionPGDLS(
#    classifier=clf_l,
#    surrogate_classifier=clf_l,
#    surrogate_data=tr_set,
#    distance=noise_type,
#    dmax=dmax,
#    lb=lb, ub=ub)

nb_attack=4000

#class from which the attack starts
class_attack=0
#class_attack=1
nb_repet=5
for rep in range(0,nb_repet):
	output_pt_attacks=np.empty([nb_attack,nb_col-1],float)
	#output_pt_attacks2=np.empty([nb_attack,nb_col-1],float)
	nb_successful_attack=np.zeros([1,1],int)
	for i in range(nb_attack):
		#take a point at random being the starting point of the attack
		idx_candidates = np.where(Y_test == class_attack)

		#select nb_init_pts points randomly in candidates and make them move
		rn = np.random.choice(idx_candidates[0].size, 1)
#	print(rn[0])
#	print(len(idx_candidates))
#	print(idx_candidates[0].size)
#	print(idx_candidates[0][rn[0]])
		x0,y0 =ts_set[idx_candidates[0][rn[0]],:].X, ts_set[idx_candidates[0][rn[0]],:].Y

		x0=x0.astype(float)
		y0=y0.astype(int)

		# Run the evasion attack on x0
		y_pred_pgd, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)
		#y_pred_pgd2, _, adv_ds_pgd2, _ = pgd_attack2.run(x0, y0)



		print("Original x0 label: ", y0.item())
		print("Adversarial example label (PGD): ", y_pred_pgd.item())
		#print("Adversarial example label (PGDLS2): ", y_pred_pgd2.item())

		print("Number of classifier gradient evaluations: {:}"
		      "".format(pgd_attack.grad_eval))

		#print("Initial sample feature values: ", x0)
		#print("Final sample(s) feature values: ", adv_ds_pgd)
		#print("Final sample(s) feature values: ", adv_ds_pgd2)


		attack_pt = adv_ds_pgd.X.tondarray()[0]
		#attack_pt2 = adv_ds_pgd2.X.tondarray()[0]
		## apply type constraints to the values if any (Boolean, between 0 and 1, etc.)
		## constraints are applied on both resulting points of attacks
		for idx in range(0,nb_col):
			if idx <= 119 :
				attack_pt[idx] = round(attack_pt[idx])
				#attack_pt2[idx] = round(attack_pt2[idx])
			if idx > 119 and idx <= 129 :
				if attack_pt[idx] < 0:
					attack_pt[idx] = 0
				if attack_pt[idx] > 1:
					attack_pt[idx] = 1

				#if attack_pt2[idx] < 0:
				#	attack_pt2[idx] = 0
				#if attack_pt2[idx] > 1:
				#	attack_pt2[idx] = 1
			if idx == 130:
				attack_pt[idx] = 0
				#attack_pt2[idx] = 0
			if idx > 130 and idx <= 144:
				if attack_pt[idx] < 0:
					attack_pt[idx] = 0
				if attack_pt[idx] > 1:
					attack_pt[idx] = 1

				#if attack_pt2[idx] < 0:
				#	attack_pt2[idx] = 0
				#if attack_pt2[idx] > 1:
				#	attack_pt2[idx] = 1
			## end constraints
	
		#check whether class has changed
		if y0.item() != y_pred_pgd.item():
			nb_successful_attack[0,0] = nb_successful_attack[0,0]+1
		#if y0.item() != y_pred_pgd2.item():
		#	nb_successful_attack[0,1] = nb_successful_attack[0,1]+1

		output_pt_attacks[i] = attack_pt
		#output_pt_attacks2[i] = attack_pt2
	
	print("End iter")

#print(output_pt_attacks)
#print(nb_successful_attack)

#if class_to_attack == 0:
#	#out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+"_label_nonacc.txt"
#	out_filename="perf_prediction_adv_config_nb_pts_"+str(nb_attack)+"_label_nonacc.txt"
#	f=open("../../../results/MOTIV/RQ1.1/secML/non_acc/"+out_filename,"w")
#else:
#	#out_filename="perf_prediction_4000_adv_config_step_"+str(stp)+"_nb_displacement_"+str(max_disp)+".txt"
#	out_filename="perf_prediction_adv_config_nb_pts_"+str(nb_attack)+"_label_acc.txt"
#	f=open("../../../results/MOTIV/RQ1.1/secML/acc/"+out_filename,"w")
#orig_stdout = sys.stdout
#sys.stdout = f

#print str(nb_succeeded_attack)
#print(dmax)
#print (nb_successful_attack)
#print ("mean: "+ str(np.mean(nb_successful_attack,axis=0)))
#print "avg: "+str(np.average(nb_succeeded_attack,axis=1))
#print ("std_dev: "+str(np.std(nb_successful_attack,axis=0)))
#print "var: "+str(np.var(nb_succeeded_attack,axis=1))

#print ("max: "+str(np.amax(nb_successful_attack,axis=0)))
#print ("min: "+str(np.amin(nb_successful_attack,axis=0)))
#print pt
#sys.stdout = orig_stdout
#f.close()

#save in file
	if y0 == 0:
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/non_acc/test_adv_attack_secML_dmax_"+str(dmax)+"_label_0_linear_rep_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/non_acc/test_adv_attack_secML_dmax_"+str(dmax)+"_label_0_Log_rep_"+str(rep)+".csv",output_pt_attacks2,delimiter=',')
		np.savetxt("../../../results/MOTIV/RQ1.1/secML/non_acc/balanced_test_adv_attack_secML_dmax_"+str(dmax)+"_label_0_linear_rep_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/non_acc/balanced_test_adv_attack_secML_dmax_"+str(dmax)+"_label_0_Log_rep_"+str(rep)+".csv",output_pt_attacks2,delimiter=',')
	else:
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/acc/test_adv_attack_secML_dmax_"+str(dmax)+"_label_1_linear_rep_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/acc/test_adv_attack_secML_dmax_"+str(dmax)+"_label_1_Log_rep_"+str(rep)+".csv",output_pt_attacks2,delimiter=',')
		np.savetxt("../../../results/MOTIV/RQ1.1/secML/acc/balanced_test_adv_attack_secML_dmax_"+str(dmax)+"_label_1_linear_rep_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/MOTIV/RQ1.1/secML/acc/balanced_test_adv_attack_secML_dmax_"+str(dmax)+"_label_1_Log_rep_"+str(rep)+".csv",output_pt_attacks2,delimiter=',')

	#np.savetxt("../../../results/MOTIV/RQ1.1/secML/check_valid/successful_secML_attacks_"+str(nb_attack)+"_pts_dmax_"+str(dmax)+"_label_"+str(y0.item())+"_rep_"+str(rep)+".csv",nb_successful_attack,delimiter=',')
	np.savetxt("../../../results/MOTIV/RQ1.1/secML/check_valid/balanced_successful_secML_attacks_"+str(nb_attack)+"_pts_dmax_"+str(dmax)+"_label_"+str(y0.item())+"_rep_"+str(rep)+".csv",nb_successful_attack,delimiter=',')

