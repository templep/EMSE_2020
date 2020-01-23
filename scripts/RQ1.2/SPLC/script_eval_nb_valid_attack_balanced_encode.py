import numpy as np

from os import path
import glob
import sys

def save(f,nb_failed):
	output_f=path.basename(f)
	#output=open("../../../results/MOTIV/4000_gen_adv_config/results_valid_attack/adv_attack_norm/"+output_f,"w")
	output=open("../../../results/MOTIV/RQ1.2/SPLC/"+output_f,"w")

	orig_stdout = sys.stdout
	sys.stdout = output
	print ("nb invalid attack: "+str(nb_failed))
	print ("nb valid attack: "+str(4000-nb_failed))

	sys.stdout = orig_stdout
	output.close()


#files=listdir("../../results/config_pt/adv_attack_norm/")


##adv attack
#filenames=glob.glob("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/acc/balanced_*20.csv")
#filenames=glob.glob("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/acc/balanced_*50.csv")
filenames=glob.glob("../../../results/MOTIV/RQ1.1/SPLC/adv_attack_norm/acc/balanced_*100.csv")
##random attack
#filenames=glob.glob("../../../results/MOTIV/RQ1.3/SPLC/random_attack_norm/acc/*20.csv")
#filenames=glob.glob("../../../results/MOTIV/RQ1.3/SPLC/random_attack_norm/acc/*50.csv")
#filenames=glob.glob("../../../results/MOTIV/RQ1.3/SPLC/random_attack_norm/acc/*100.csv")
#print (filenames)


for f in filenames:
	nb_failed=0

	#load data
	data=np.genfromtxt(f,delimiter=",")
	data_attack=data[7275:,:]

	nb_pts = len(data_attack)

	#check that mutual exclusion are satisfied: only one category in the first 10 features is selected each time
	#the rest of the features are forced to meet boundaries after displacements
	res = np.where(data_attack[:,0:15] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	data_decode = np.reshape(feat_decode,[-1,1])
	#print(np.shape(data_decode))
	#print(data_decode)

	res = np.where(data_attack[:,16:23] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,24:39] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,40:47] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,48:63] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,64:71] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,72:87] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,88:95] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,96:111] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)

	res = np.where(data_attack[:,112:119] == 1)
	feat_decode = res[1]
	if nb_pts != len(feat_decode):
		nb_f = len(feat_decode) - nb_pts
		nb_failed = max(nb_failed,nb_f)
		
#
#	for l in data_attack:
#		if (l[0] > 16) or (l[1] > 8) or (l[2] > 16) or (l[3] > 8) or (l[4] > 16) or (l[5] > 8) or (l[6] > 16) or (l[7] > 8) or (l[8] > 16) or (l[9] > 8) or (l[10] < 0) or (l[10] > 1) or (l[11] < 0) or (l[11] > 1) or (l[12] < 0) or (l[12] > 1) or (l[13] < 0) or (l[13] > 1) or (l[14] < 0) or (l[14] > 1) or (l[15] < 0) or (l[15] > 1) or (l[16] < 0) or (l[16] > 1) or (l[17] < 0) or (l[17] > 1) or (l[18] < 0) or (l[18] > 1) or (l[19] < 0) or (l[19] > 1) or (l[20] < 0) or (l[20] > 1) or (l[21] < 0) or (l[21] > 1) or (l[22] < 0) or (l[22] > 1) or (l[29] < 0) or (l[29] > 1) or (l[30] < 0) or (l[30] > 1) or (l[31] < 0) or (l[31] > 1) or (l[32] < 0) or (l[32] > 1) or (l[33] < 0) or (l[33] > 1) or (l[34] < 0) or (l[34] > 1):
#			nb_failed +=1

	save(f,nb_failed)
