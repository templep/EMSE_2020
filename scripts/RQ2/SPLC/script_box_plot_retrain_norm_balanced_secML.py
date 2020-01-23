import numpy as np
import matplotlib.pyplot as plt

#f_ori="./perf_classifier_25_attacks_norm.csv"
f_ori = "../../../results/MOTIV/RQ2/secML/acc_after_retrain_attacks_25_pts_label_1.csv"
#f1="./result_perf_after_retrain_25_pts_balanced.csv"
f1 = "../../../results/MOTIV/RQ2/secML/balanced_acc_after_retrain_attacks_25_pts_label_1.csv"

#f_ori is the file containing original results (without any balance nor data augmentation) -> classif perf over test set: 96.4562
#f1 contains classification performances when the training set only is balanced -> classif perf over test set: 94.9588
#f2 contains classification performances when both the training set and the test set are balanced -> classif perf over test set: 96.7143

data=np.genfromtxt(f_ori,delimiter=",")
data=data*100
cst = np.mean(data[0,:])
data = np.delete(data,[0],axis=0)
data_t = data

data2=np.genfromtxt(f1,delimiter=",")
data2=data2*100
cst2 = np.mean(data2[0,:])
data2 = np.delete(data2,[0],axis=0)
data_t2 = data2

nb_elem = 5

##first data set with balanced training set only
fig, ax = plt.subplots()

ax.set_ylim(85,100)
#plt.setp(ax,xticks=np.linspace(0,1,num=7,endpoint=True),xticklabels=np.logspace(-6,6,num=7,endpoint=True))
plt.xticks([0,1,2,3,4],labels = ['0.1', '0.5', '1.0', '5.0', '10.0'])


#plot original perf (f_ori)
offset = -0.2
edge_color="tomato"
fill_color="white"
pos = np.arange(data_t.shape[1])+offset
bp1 = ax.boxplot(data_t, labels = ['0.1', '0.5', '1.0', '5.0', '10.0'], positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp1[element], color=edge_color)
for patch in bp1['boxes']:
    patch.set(facecolor=fill_color)

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*96.4562,'-r')
plt.plot([0,1,2,3,4],np.ones(nb_elem)*cst,'-r')

#plot perf with balanced training set only (f1)
offset = +0.2
edge_color="skyblue"
fill_color="white"
pos = np.arange(data_t2.shape[1])+offset
bp2 = ax.boxplot(data_t2, labels = ['0.1', '0.5', '1.0', '5.0', '10.0'], positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp2[element], color=edge_color)
for patch in bp2['boxes']:
    patch.set(facecolor=fill_color)

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*94.9588,'-b')
plt.plot([0,1,2,3,4],np.ones(nb_elem)*cst2,'-b')


#set the labels and legends
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],['without balance','with balance'],loc='lower right')
ax.set_xlabel("d_max values")
ax.set_ylabel("Accuracy of classifier")

#plt.show()
from os import path
plt.savefig("../../../results/MOTIV/RQ2/secML/result_perf_after_retrain_25_pts_balanced.png", bbox_inches='tight')
plt.close()




