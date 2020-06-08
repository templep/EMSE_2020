import numpy as np
from scipy import stats
import pandas as pd

##random data
##acceptable
balanced_random_data = pd.read_csv("../../../../../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_20_disp.csv",header=None,skiprows=0)
#balanced_random_data = pd.read_csv("../../../../../results/JHipster/RQ1.3/SPLC/result_valid_misclf_random_100_disp.csv",header=None,skiprows=0)


##SPLC
##acceptable
balanced_data = pd.read_csv("../../../../../results/JHipster/RQ1.2/result_valid_misclf_SPLC_20_disp.csv",header=None,skiprows=0)
#balanced_data = pd.read_csv("../../../../../results/JHipster/RQ1.2/result_valid_misclf_SPLC_100_disp.csv",header=None,skiprows=0)


t_stat,p_val = stats.ttest_ind(balanced_random_data,balanced_data,equal_var=False,axis=1)

#acceptable
np.savetxt("../../../../../results/stats/RQ1.3/JHipster/SPLC/result_t_test_balanced_SPLC_VS_random_20_disp.csv",[t_stat.T,p_val.T],delimiter=',')
#np.savetxt("../../../../../results/stats/RQ1.3/JHipster/SPLC/result_t_test_balanced_SPLC_VS_random_100_disp.csv",[t_stat.T,p_val.T],delimiter=',')
