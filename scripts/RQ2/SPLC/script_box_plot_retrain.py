import numpy as np
import matplotlib.pyplot as plt

f="../changing_params/expe_final_retrain_python/perf_classifier_25_attacks_norm.csv"

nb_elem = 6
base=96.4562
cst = np.ones(nb_elem)*base

data=np.genfromtxt(f,delimiter=",")

fig, ax = plt.subplots()

ax.boxplot(data, labels = np.logspace(-4,1,num=nb_elem,endpoint=True))
ax.plot(np.logspace(-4,1,num=nb_elem,endpoint=True), cst)

ax.set_ylim(80,100)

#ax.set_xlabel("displacement step size")
#ax.set_ylabel("Accuracy of the classifier after retraining (25 adversarial configurations)")

plt.show()



