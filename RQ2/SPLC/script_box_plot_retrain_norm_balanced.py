import numpy as np
import matplotlib.pyplot as plt

#f_ori="./perf_classifier_25_attacks_norm.csv"
f_ori = "../../../results/MOTIV/RQ2/SPLC/acc_after_retrain_attacks_25_pts_20_disp_label_1.csv"
#f1="./result_perf_after_retrain_25_pts_balanced.csv"
f1 = "../../../results/MOTIV/RQ2/SPLC/acc_balanced_after_retrain_attacks_25_pts_20_disp_label_1.csv"
#f2="./result_perf_after_retrain_25_pts_balanced_augmented.csv"
f2 = "../../../results/MOTIV/RQ2/SPLC/acc_balanced_after_retrain_attacks_25_pts_20_disp_label_1.csv"

#f_ori is the file containing original results (without any balance nor data augmentation) -> classif perf over test set: 96.4562
#f1 contains classification performances when the training set only is balanced -> classif perf over test set: 94.9588
#f2 contains classification performances when both the training set and the test set are balanced -> classif perf over test set: 96.7143

data=np.genfromtxt(f_ori,delimiter=",")
#data_t = np.transpose(data)
data_t = data * 100

data2=np.genfromtxt(f1,delimiter=",")
#data_t2 = np.transpose(data2)
data_t2 = data2 * 100


data3=np.genfromtxt(f2,delimiter=",")
#data_t3 = np.transpose(data3)
data_t3 = data3 * 100


##first data set with balanced training set only
fig, ax = plt.subplots()

ax.set_ylim(85,100)
#plt.setp(ax,xticks=np.linspace(0,1,num=7,endpoint=True),xticklabels=np.logspace(-6,6,num=7,endpoint=True))
plt.xticks([0,1,2,3,4,5,6],np.logspace(-4,2,num=7,endpoint=True))


#plot original perf (f_ori)
offset = -0.2
edge_color="tomato"
fill_color="white"
pos = np.arange(data_t.shape[1])+offset
bp1 = ax.boxplot(data_t, positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp1[element], color=edge_color)
for patch in bp1['boxes']:
    patch.set(facecolor=fill_color)

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*96.4562,'-r')
plt.plot([0,1,2,3,4,5,6],np.ones(7)*90.7616,'-r')

#plot perf with balanced training set only (f1)
offset = +0.2
edge_color="skyblue"
fill_color="white"
pos = np.arange(data_t2.shape[1])+offset
bp2 = ax.boxplot(data_t2, positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp2[element], color=edge_color)
for patch in bp2['boxes']:
    patch.set(facecolor=fill_color)

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*94.9588,'-b')
plt.plot([0,1,2,3,4,5,6],np.ones(7)*92.3151,'-b')


#set the labels and legends
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],['without balance','with balance (training)'],loc='lower right')
ax.set_xlabel("displacement step size")
ax.set_ylabel("Accuracy of classifier")

#plt.show()
from os import path
base=path.basename(f2)
plt.savefig("../../../results/MOTIV/RQ2/SPLC/result_perf_after_retrain_25_pts_balanced.png", bbox_inches='tight')
plt.close()


##second plot with augmented data sets
fig, ax = plt.subplots()

ax.set_ylim(87,100)
plt.xticks([0,1,2,3,4,5,6],np.logspace(-4,2,num=7,endpoint=True))

#plot original perf (f_ori)
offset = -0.2
edge_color="tomato"
fill_color="white"
pos = np.arange(data_t.shape[1])+offset
bp1 = ax.boxplot(data_t, positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp1[element], color=edge_color)
for patch in bp1['boxes']:
    patch.set(facecolor=fill_color)

#p1=[0,96.4562]
#p2=[6,96.4562]
p1=[0,90.7616]
p2=[6,90.7616]

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*96.4562,'-r')
plt.plot([0,1,2,3,4,5,6],np.ones(7)*90.7616,'-r')

#plot perf with both data sets balanced (f2)
offset = +0.2
edge_color="skyblue"
fill_color="white"
pos = np.arange(data_t3.shape[1])+offset
bp2 = ax.boxplot(data_t3, positions= pos, widths=0.3, patch_artist=True)#, manage_ticks=False)
for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp2[element], color=edge_color)
for patch in bp2['boxes']:
    patch.set(facecolor=fill_color)

#p1=[0,96.7143]
#p2=[6,96.7143]
p1=[0,92.3151]
p2=[6,92.3151]

#plt.plot([0,1,2,3,4,5,6],np.ones(7)*96.7143,'-b')
plt.plot([0,1,2,3,4,5,6],np.ones(7)*92.3151,'-b')

plt.xticks([0,1,2,3,4,5,6],np.logspace(-6,6,num=7,endpoint=True))
ax.legend([bp1["boxes"][0], bp2["boxes"][0]],['without balance','with balance'],loc='lower right')

ax.set_xlabel("displacement step size")
ax.set_ylabel("Accuracy of classifier")

from os import path
base=path.basename(f2)
plt.savefig("../../../results/MOTIV/RQ2/SPLC/result_perf_after_retrain_25_pts_balanced_augmented.png", bbox_inches='tight')
plt.close()



