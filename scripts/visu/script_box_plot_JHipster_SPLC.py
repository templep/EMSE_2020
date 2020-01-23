import numpy as np
import matplotlib.pyplot as plt

#### class acceptable attacked
##adv attacks
#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_20_disp.csv"
#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_50_disp.csv"
#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_100_disp.csv"

#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_20_disp_non_acc.csv"
#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_50_disp_non_acc.csv"
#f="../results/JHipster/RQ1.2/result_valid_misclf_SPLC_100_disp_non_acc.csv"

f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_20_disp.csv"
#f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_50_disp.csv"
#f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_100_disp.csv"

#f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_20_disp_non_acc.csv"
#f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_50_disp_non_acc.csv"
#f="../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_100_disp_non_acc.csv"


data=np.genfromtxt(f,delimiter=",")
data_t = np.transpose(data)

fig, ax = plt.subplots()

ax.boxplot(data_t, labels = np.logspace(-6,6,num=7,endpoint=True))

ax.set_ylim(-10,1050)

ax.set_xlabel("displacement step size")
ax.set_ylabel("Number of successful attacks (over 4000)")

plt.show()



