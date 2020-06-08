import numpy as np
from scipy import stats
import pandas as pd


##acceptable
#balanced data
balanced_data = pd.read_csv("../../../../../results/MOTIV/RQ2/SPLC/acc_balanced_after_retrain_attacks_25_pts_20_disp_label_1.csv",header=None,skiprows=0)
balanced_data = balanced_data.T

##baseline
##balanced data
baseline = np.ones(balanced_data.shape)
baseline = baseline*92.3151
baseline_df = pd.DataFrame(data=baseline)

t_stat,p_val = stats.ttest_ind(balanced_data, baseline,equal_var=False,axis=1)

np.savetxt("../../../../../results/stats/RQ2/MOTIV/SPLC/result_t_test_balanced_VS_baseline.csv",[t_stat.T,p_val.T],delimiter=',')

##acceptable
##unbalanced data
unbalanced_data = pd.read_csv("../../../../../results/MOTIV/RQ2/SPLC/acc_after_retrain_attacks_25_pts_20_disp_label_1.csv",header=None,skiprows=0)
unbalanced_data = unbalanced_data.T

##baseline
##unbalanced data
baseline = np.ones(unbalanced_data.shape)
baseline = baseline*90.7616
baseline_df = pd.DataFrame(data=baseline)

t_stat,p_val = stats.ttest_ind(unbalanced_data,baseline,equal_var=False,axis=1)

##unbalanced data
np.savetxt("../../../../../results/stats/RQ2/MOTIV/SPLC/result_t_test_unbalanced_VS_baseline.csv",[t_stat.T,p_val.T],delimiter=',')
