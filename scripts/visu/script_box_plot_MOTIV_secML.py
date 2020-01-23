import numpy as np
import matplotlib.pyplot as plt

#def of input file
#f = "../results/MOTIV/RQ1.2/secML/result_valid_misclf_SecML_acc.csv"
#f = "../results/MOTIV/RQ1.2/secML/result_valid_misclf_SecML_non_acc.csv"
#f = "../results/MOTIV/RQ1.2/secML/result_balanced_valid_misclf_SecML_acc.csv"
f = "../results/MOTIV/RQ1.2/secML/result_balanced_valid_misclf_SecML_non_acc.csv"

data=np.genfromtxt(f,delimiter=",")

#take only two first lines
data = data[0:5,:]
data_t = np.transpose(data)

fig, ax = plt.subplots()

#ax.boxplot(data_t, labels = np.logspace(-6,6,num=7,endpoint=True))
ax.boxplot(data_t, labels = ['0.1','0.5','1.0','5.0','10.0'])

ax.set_ylim(-50,4050)

ax.set_xlabel("d_max values")
ax.set_ylabel("Number of successful attacks (over 4000)")

plt.show()



