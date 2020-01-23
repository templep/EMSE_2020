import numpy as np
import matplotlib.pyplot as plt

f="../../../results/JHipster/RQ2/SPLC/acc_retrain_adv_attack_nb_displacement_20_label_1.csv"


data=np.genfromtxt(f,delimiter=",")
data=data*100

nb_elem = 6
base=data[0,:]
data = np.delete(data,[0],axis=0)
print(data)
cst = np.ones(nb_elem)*base

fig, ax = plt.subplots()

ax.boxplot(data, labels = np.logspace(-4,1,num=nb_elem,endpoint=True))
ax.plot(np.logspace(-4,1,num=nb_elem,endpoint=True), cst)

ax.set_ylim(92,100)

#ax.set_xlabel("displacement step size")
#ax.set_ylabel("Accuracy of the classifier after retraining (25 adversarial configurations)")

plt.show()



