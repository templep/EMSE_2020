import numpy as np
from scipy import stats
import pandas as pd

##balanced data
##acceptable
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_20_disp.csv",header=None,skiprows=0)
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_50_disp.csv",header=None,skiprows=0)
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_100_disp.csv",header=None,skiprows=0)

##non-acceptable
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_20_disp_non_acc.csv",header=None,skiprows=0)
#balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_50_disp_non_acc.csv",header=None,skiprows=0)
balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_balanced_valid_misclf_SPLC_100_disp_non_acc.csv",header=None,skiprows=0)

##not balanced data
##acceptable
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_20_disp.csv",header=None,skiprows=0)
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_50_disp.csv",header=None,skiprows=0)
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_100_disp.csv",header=None,skiprows=0)

##non-acceptable
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_20_disp_non_acc.csv",header=None,skiprows=0)
#unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_50_disp_non_acc.csv",header=None,skiprows=0)
unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ1.2/SPLC/result_valid_misclf_SPLC_100_disp_non_acc.csv",header=None,skiprows=0)


t_stat,p_val = stats.ttest_ind(balanced_data,unbalanced_data,equal_var=False,axis=1)

#acceptable
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_20_disp.csv",[t_stat.T,p_val.T],delimiter=',')
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_50_disp.csv",[t_stat.T,p_val.T],delimiter=',')
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_100_disp.csv",[t_stat.T,p_val.T],delimiter=',')

#non-acceptable
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_20_disp_non_acc.csv",[t_stat.T,p_val.T],delimiter=',')
#np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_50_disp_non_acc.csv",[t_stat.T,p_val.T],delimiter=',')
np.savetxt("../../../../../results/stats/RQ1.1/MOTIV/SPLC/result_t_test_balanced_VS_unbalanced_100_disp_non_acc.csv",[t_stat.T,p_val.T],delimiter=',')
