import numpy as np

from os import path
import glob
import sys


##adv attack
#filenames=glob.glob("../../../results/MOTIV/config_pt/adv_attack_norm/non_acc/balanced_*_20_label_nonacc.csv")
#filenames=glob.glob("../../../results/MOTIV/config_pt/adv_attack_norm/non_acc/balanced_*50_label_nonacc.csv")
#filenames=glob.glob("../../../results/MOTIV/config_pt/adv_attack_norm/non_acc/balanced_*100_label_nonacc.csv")
##random attack
#filenames=glob.glob("../../../results/MOTIV/config_pt/random_attack_norm/non_acc/balanced_*20_nonacc.csv")
#filenames=glob.glob("../../../results/MOTIV/config_pt/random_attack_norm/non_acc/balanced_*50_nonacc.csv")
filenames=glob.glob("../../../results/MOTIV/config_pt/random_attack_norm/non_acc/balanced_*100_nonacc.csv")
#print (filenames)

for f in filenames:
	nb_failed=0

	data=np.genfromtxt(f,delimiter=",")
	data_attack=data[489:,:]

	for l in data_attack:
		if (l[0] > 16) or (l[1] > 8) or (l[2] > 16) or (l[3] > 8) or (l[4] > 16) or (l[5] > 8) or (l[6] > 16) or (l[7] > 8) or (l[8] > 16) or (l[9] > 8) or (l[10] < 0) or (l[10] > 1) or (l[11] < 0) or (l[11] > 1) or (l[12] < 0) or (l[12] > 1) or (l[13] < 0) or (l[13] > 1) or (l[14] < 0) or (l[14] > 1) or (l[15] < 0) or (l[15] > 1) or (l[16] < 0) or (l[16] > 1) or (l[17] < 0) or (l[17] > 1) or (l[18] < 0) or (l[18] > 1) or (l[19] < 0) or (l[19] > 1) or (l[20] < 0) or (l[20] > 1) or (l[21] < 0) or (l[21] > 1) or (l[22] < 0) or (l[22] > 1) or (l[29] < 0) or (l[29] > 1) or (l[30] < 0) or (l[30] > 1) or (l[31] < 0) or (l[31] > 1) or (l[32] < 0) or (l[32] > 1) or (l[33] < 0) or (l[33] > 1) or (l[34] < 0) or (l[34] > 1):
			nb_failed +=1

	output_f=path.basename(f)
	#output=open("../../../results/MOTIV/4000_gen_adv_config/results_valid_attack/adv_attack_norm/"+output_f,"w")
	output=open("../../../results/MOTIV/4000_gen_adv_config/results_valid_attack/random_attack_norm/"+output_f,"w")

	orig_stdout = sys.stdout
	sys.stdout = output
	print ("nb invalid attack: "+str(nb_failed))
	print ("nb valid attack: "+str(4000-nb_failed))

	sys.stdout = orig_stdout
	output.close()
