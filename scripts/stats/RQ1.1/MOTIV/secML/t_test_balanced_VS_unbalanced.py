import numpy as np
from scipy import stats
import pandas as pd

##balanced data
##acceptable
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/secML/result_balanced_valid_misclf_SecML_acc.csv",header=None,skiprows=0)

##non-acceptable
balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/secML/result_balanced_valid_misclf_SecML_non_acc.csv",header=None,skiprows=0)

##not balanced data
##acceptable
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/secML/result_valid_misclf_SecML_acc.csv",header=None,skiprows=0)

##non-acceptable
unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/secML/result_valid_misclf_SecML_non_acc.csv",header=None,skiprows=0)


t_stat,p_val = stats.ttest_ind(balanced_data,unbalanced_data,equal_var=False,axis=1)

#acceptable
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/secML/result_t_test_balanced_VS_unbalanced_acc.csv",[t_stat.T,p_val.T],delimiter=',')

#non-acceptable
np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/secML/result_t_test_balanced_VS_unbalanced_non_acc.csv",[t_stat.T,p_val.T],delimiter=',')
