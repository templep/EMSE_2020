import pandas as pd
import numpy as np
import sys


input_path="../../../data/JHipster/data_clean/"
input_filename="jhipster_transformed_ready_build.csv"
input_filename_smp="data_randomly_sampled_200_configurations.csv"
input_file=input_path+input_filename
input_file_smp=input_path+input_filename_smp

#raw_data= pd.read_csv(input_file,dtype=int)
#data_smp= pd.read_csv(input_file_smp,dtype=int)
raw_data = np.loadtxt(input_file, delimiter=',',skiprows=1)
data_smp = np.loadtxt(input_file_smp, delimiter=',',skiprows=1)

#column names
#column_names = raw_data.columns.tolist()

#test nb lines and columns
nb_line = len(raw_data)
nb_col = raw_data[1].size

print(nb_line)
print(nb_col)

#print(raw_data.dtypes)


if(nb_line != 90209):
	sys.exit("expected number of lines is 90209 while we retrieved "+str(nb_line))

if(nb_col != 47):
	sys.exit("expected number of cols is 47 while we retrieved "+str(nb_col))


##separated class from the rests of the features
Y_full=raw_data[:,nb_col-1]
Y_full.astype(int)
#del raw_data[:,nb_col-1]
raw_data=np.delete(raw_data,nb_col-1,axis=1)
Y_smp=data_smp[:,nb_col-1]
Y_smp.astype(int)
#del data_smp[:,nb_col-1]
data_smp=np.delete(data_smp,nb_col-1,axis=1)
#data_smp.drop(columns=str(nb_col-1))

print("End preprocessing -> separating class and other features")

#remove data from data_smp (sample used as training set) from the set of all data points
import numpy as np
for x in data_smp:
	idx = np.where((raw_data == x.T).all(axis=1))
	raw_data = np.delete(raw_data,idx,axis=0)
	Y_full = np.delete(Y_full,idx,axis=0)

import sys
if(len(raw_data) > nb_line-len(data_smp)):
	print("Expected number of lines:"+str(nb_line-len(data_smp)))
	print("Retrieved number of lines:"+str(len(raw_data)))
	sys.exit("Expected number of line in test set is not matched")

#create associate dataFrame and convert into secML structure
df_raw_data = pd.DataFrame(raw_data)
df_data_smp = pd.DataFrame(data_smp)
df_Y = pd.DataFrame(Y_full)
df_Y_smp = pd.DataFrame(Y_smp)

from secml.data import CDataset
raw_data_encoded_secML = CDataset(df_raw_data,df_Y)
data_smp_encoded_secML = CDataset(df_data_smp,df_Y_smp)

print("SecML dataset created")

#split between train and test -> train = 66% of data
n_tr = data_smp_encoded_secML.num_samples
n_ts = raw_data_encoded_secML.num_samples

#from secml.data.splitter import CTrainTestSplit
#splitter = CTrainTestSplit(
#    train_size=n_tr, test_size=n_ts)
#tr, ts = splitter.split(raw_data_encoded_secML)

print("Data splitted into train and test")


# create a splitter for training data -> training and validation sets
from secml.data.splitter import CDataSplitterKFold
#default splitter for cross-val -> num_folds=3 and random_state=None
xval_splitter = CDataSplitterKFold()


##train a classifier (SVM)
from secml.ml.classifiers import CClassifierSVM
clf_lin = CClassifierSVM()

#from secml.ml.kernel import CKernelRBF
#clf_rbf = CClassifierSVM(kernel=CKernelRBF())

#problem seems linearly decidable -> try a logistic regression classifier without any parameter estimations
from secml.ml.classifiers import CClassifierLogistic
#clf_l= CClassifierLogistic()

xval_lin_params = {'C': [0.01, 0.1, 1, 10, 100]}


# Select and set the best training parameters for the linear classifier
print("Estimating the best training parameters for linear kernel...")
best_lin_params = clf_lin.estimate_parameters(
    dataset=data_smp_encoded_secML,
    parameters=xval_lin_params,
    splitter=xval_splitter,
    metric='accuracy',
    perf_evaluator='xval'
)

# Select and set the best training parameters for the RBF classifier
#print("Estimating the best training parameters for RBF kernel...")
#best_rbf_params = clf_rbf.estimate_parameters(
#    dataset=tr,
#    parameters=xval_rbf_params,
#    splitter=xval_splitter,
#    metric='accuracy',
#    perf_evaluator='xval'
#)
print(best_lin_params)

#train classifier
print("start training")
clf_lin.fit(data_smp_encoded_secML)
#print("linear training ended, begining rbf")
#clf_rbf.fit(tr)
#print("start linear classif")
#clf_l.fit(data_smp_encoded_secML)

print("Classifiers trained")


# Metric to use for training and performance evaluation
from secml.ml.peval.metrics import CMetricAccuracy
metric = CMetricAccuracy()


# Compute predictions on a test set
y_lin_pred = clf_lin.predict(raw_data_encoded_secML.X)
#y_rbf_pred = clf_rbf.predict(ts.X)
#y_l_pred = clf_l.predict(raw_data_encoded_secML.X)

# Evaluate the accuracy of the classifier
acc_lin = metric.performance_score(y_true=raw_data_encoded_secML.Y, y_pred=y_lin_pred)
#acc_rbf = metric.performance_score(y_true=ts.Y, y_pred=y_rbf_pred)
#acc_rbf = 0.0
#acc_l = metric.performance_score(y_true=raw_data_encoded_secML.Y, y_pred=y_l_pred)

print("Performance evaluations ended:")
print(acc_lin)
#print(acc_rbf)
#print(acc_l)

print("Begin setup for attack")

import random
from secml.adv.attacks.evasion import CAttackEvasionPGD
#perform adversarial attacks
noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
#dmax = 20  # Maximum perturbation
dmax = 0.1  # Maximum perturbation
lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for unbounded

solver_params = {'eta' : 0.01}

#set lower bound and upper bound respectively to 0 and 1 since all features are Boolean
pgd_attack = CAttackEvasionPGD(
    classifier=clf_lin,
    surrogate_classifier=clf_lin,
    surrogate_data=data_smp_encoded_secML,
    distance=noise_type,
    dmax=dmax,
    lb=lb, ub=ub,
    solver_params=solver_params)

#pgd_attack2 = CAttackEvasionPGD(
#    classifier=clf_l,
#    surrogate_classifier=clf_l,
#    surrogate_data=data_smp_encoded_secML,
#    distance=noise_type,
#    dmax=dmax,
#    lb=lb, ub=ub)

nb_repet = 10
nb_attack=25
output_pt_attacks=np.empty([nb_attack,nb_col-1],int)
#output_pt_attacks2=np.empty([nb_attack,nb_col-1],int)
nb_successful_attack=np.zeros([nb_repet,1],int)

#class_to_attack=0
class_to_attack=1

acc_attack = np.zeros([nb_repet+1,1],float)
acc_attack[0,0] = acc_lin

for rep in range(0,nb_repet):
	for i in range(nb_attack):
		#take a point at random being the starting point of the attack
		idx_candidates = np.where(Y_full == class_to_attack)

		#select nb_init_pts points randomly in candidates and make them move
		rn = np.random.choice(idx_candidates[0].size, 1)
		print(rn[0])
		print(len(idx_candidates))
		print(idx_candidates[0].size)
		print(idx_candidates[0][rn[0]])
		x0,y0 =raw_data_encoded_secML[idx_candidates[0][rn[0]],:].X, raw_data_encoded_secML[idx_candidates[0][rn[0]],:].Y

		x0=x0.astype(float)
		#print(x0.dtype)
		#print(y0)
		y0=y0.astype(int)
		#print(y0.dtype)

		# Run the evasion attack on x0
		y_pred_pgd, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)
		#y_pred_pgd2, _, adv_ds_pgd2, _ = pgd_attack2.run(x0, y0)



		print("Original x0 label: ", y0.item())
		print("Adversarial example label (PGD): ", y_pred_pgd.item())
		#print("Adversarial example label (PGD2): ", y_pred_pgd2.item())

		print("Number of classifier gradient evaluations: {:}"
		      "".format(pgd_attack.grad_eval))

		print("Initial sample feature values: ", x0)
		print("Final sample(s) feature values: ", adv_ds_pgd)
		#print("Final sample(s) feature values: ", adv_ds_pgd2)

	#output_path="../results/SPLC_script/"
	#output_filename="evasion_attack_points_linear_test.csv"
	#adv_ds_pgd.save(output_path+output_filename)
		attack_pt = adv_ds_pgd.X.tondarray()[0]
		print(attack_pt.shape)
		attack_pt = attack_pt.astype(int)
		#make all feature Boolean => values ={0 or 1}	
		for d in attack_pt:
			if d < 0:
				d = 0
			if d > 1:
				d = 1
		output_pt_attacks[i] = attack_pt

		#attack_pt2 = adv_ds_pgd2.X.tondarray()[0]
		#attack_pt2 = attack_pt2.astype(int)
		#for d in attack_pt2:
		#	if d < 0:
				#d = 0
		#	if d > 1:
		#		d = 1
		#output_pt_attacks2[i] = attack_pt2
#	print(x0.tondarray()[0])

	#check and force the validity wrt the FM
	for x in output_pt_attacks:
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

#for x in output_pt_attacks2:
#	if x[2] == 1:
#		if x[6] == 0 and x[5] == 0:
#			x[6] = 1
#			#could be x[5] = 1 also
#		if x[10] == 1:
#			x[10] = 0
#	if x[0] == 1 or x[1] == 1:
#		if x[7] == 0 and x[10] == 0:
#			x[7] = 1
#			#could be x[10] = 1 also
#	if x[1] == 0 and (x[3] == 0 or x[0] == 0):
#		x[20] = 0
#	if x[8] == 1:
#		if x[19] == 0 and x[21] == 0:
#			x[21] = 1
#			#could be x[19] = 1 also
#	if x[21] == 1:
#		x[36] = 0
#	if x[21] == 1 and x[32] == 1:
#		if x[23] == 0 and x[24] == 0 and x[25] == 0:
#			x[23] =1
#			#could be x[24] = 1 or x[25] =1 also
#	if x[21] == 1 and x[34] == 1:
#		if x[23] == 0 and x[24] == 0 and x[30] == 0:
#			x[23] =1
#			#could be x[24] = 1 or x[30] =1 also
#	if x[21] == 1 and x[35] == 1:
#		if x[23] == 0 and x[24] == 0 and x[27] == 0:
#			x[23] =1
#			#could be x[24] = 1 or x[27] =1 also
#	if x[21] == 1 and x[37] == 1:
#		if x[23] == 0 and x[24] == 0 and x[28] == 0:
#			x[23] =1
#			#could be x[24] = 1 or x[28] =1 also
#	if x[5] == 1 and x[10] == 0:
#		x[7] = 1
#	if x[18] == 1 or x[19] == 1 or x[20] == 1:
#		x[29] = 1
#		x[36] = 1
#	if x[3] == 1:
#		x[10] = 1
#	if (x[19] == 1 or x[18] == 1) and (x[0] == 0 or x[20] == 1):
#		x[14] = 1
#	if x[0] == 1:
#		x[12] = 1
#	if x[18] == 1 or x[2] == 0 or (x[9] == 0 and x[7] == 0):
#		x[40] = 0
#	if x[21] == 0:
#		x[39] = 0
#	if (x[2] == 0 and x[0] == 0) or (x[20] == 0 and x[12] == 0):
#		x[15] = 0
#	if (x[2] == 0 and x[0] == 0):
#		x[16] = 0
#	if (x[1] == 1 or x[3] == 1):
#		x[44] = 1
#		x[45] = 0
#	if x[0] == 1 or x[2] == 1:
#		x[45] = 1
#		x[42] = 1
#		#could be x[43] = 1 also
#
		#check whether class has changed
		if y0.item() != y_pred_pgd.item():
			nb_successful_attack[rep,0] = nb_successful_attack[rep,0]+1
		#if y0.item() != y_pred_pgd2.item():
			#nb_successful_attack[rep,1] = nb_successful_attack[0,1]+1

	print("End")

	#save in file
	if y0 == 0:
		np.savetxt("../../../results/JHipster/RQ2/secML/non_acc/test_adv_attack_secML_dmax_"+str(dmax)+"_label_0_linear_repet_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/JHipster/RQ2/secML/non_acc/test_adv_attack_secML_dmax_"+str(dmax)+"label_0_Log.csv",output_pt_attacks2,delimiter=',')
	else:
		np.savetxt("../../../results/JHipster/RQ2/secML/acc/test_adv_attack_secML_dmax_"+str(dmax)+"label_1_linear_repet_"+str(rep)+".csv",output_pt_attacks,delimiter=',')
		#np.savetxt("../../../results/JHipster/RQ2/secML/acc/test_adv_attack_secML_dmax_"+str(dmax)+"label_1_Log.csv",output_pt_attacks2,delimiter=',')

	np.savetxt("../../../results/JHipster/RQ2/secML/check_valid/successful_secML_attacks_"+str(nb_attack)+"_pts_dmax_"+str(dmax)+"_label_"+str(y0.item())+"_repet_"+str(rep)+".csv",nb_successful_attack,delimiter=',')

#for j in range(0, nb_col-1, 1):
#	if(x0[j] != adv_ds_pgdls.X[j]):
#		print("feature {:}".format(j)+":"+"{:}".format(x0[j])+"\t{:}".format(adv_ds_pgdls.X[j]))
#print("Comparison ended")


	##retraining
	print("BEGIN RETRAINING")
	class_to_add = np.ones((nb_attack,1),dtype = int)
	class_attack_pt = class_to_add*class_to_attack
	Y_to_add = np.transpose(class_attack_pt)[0]
	to_add = CDataset(output_pt_attacks,Y_to_add)
	tr_set_add = data_smp_encoded_secML.append(to_add)

#class_to_add = np.ones((nb_attack,1),dtype = int)
#class_attack_pt = class_to_add*class_to_attack
#Y_to_add = np.transpose(class_attack_pt)[0]
#to_add = CDataset(output_pt_attacks2,Y_to_add)
#tr_set_add2 = data_smp_encoded_secML.append(to_add)

	print("param estimation")
	best_lin_params = clf_lin.estimate_parameters(
	    dataset=tr_set_add,
	    parameters=xval_lin_params,
	    splitter=xval_splitter,
	    metric='accuracy',
	    perf_evaluator='xval'
	)

#best_lin_params = clf_l.estimate_parameters(
#    dataset=tr_set_add2,
#    parameters=xval_lin_params,
#    splitter=xval_splitter,
#    metric='accuracy',
#    perf_evaluator='xval'
#)

	clf_lin.fit(tr_set_add)
	#clf_l.fit(tr_set_add2)

# Compute predictions on a test set
	y_pred = clf_lin.predict(raw_data_encoded_secML.X)
	#y_pred2 = clf_l.predict(raw_data_encoded_secML.X)

# Evaluate the accuracy of the classifier
	acc = metric.performance_score(y_true=raw_data_encoded_secML.Y, y_pred=y_pred)
	#acc2 = metric.performance_score(y_true=raw_data_encoded_secML.Y, y_pred=y_pred2)

	acc_attack[rep+1,0] = acc

	print("Accuracy on test set after retraining: {:.2%}".format(acc))
	#print("Accuracy on test set after retraining2: {:.2%}".format(acc2))

	np.savetxt("../../../results/JHipster/RQ2/secML/acc_after_retrain_attacks_"+str(nb_attack)+"_pts_dmax_"+str(dmax)+"_label_"+str(y0.item())+"_repet_"+str(rep)+".csv",acc_attack,delimiter=',')


