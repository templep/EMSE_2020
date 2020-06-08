import numpy as np
from scipy import stats
import pandas as pd


##acceptable
#balanced data
balanced_data = pd.read_csv("../../../../../results/JHipster/RQ2/secML/acc_after_retrain_attacks_25_pts_label_1.csv",header=None,skiprows=0)
balanced_data = balanced_data.T

##baseline
cst = np.mean(balanced_data[:1].to_numpy())
balanced_data=balanced_data.iloc[1:]
baseline = np.ones(balanced_data.shape)
baseline = baseline*cst
baseline_df = pd.DataFrame(data=baseline)

t_stat,p_val = stats.ttest_ind(balanced_data, baseline,equal_var=False,axis=1)

np.savetxt("../../../../../results/stats/RQ2/JHipster/secML/result_t_test_balanced_VS_baseline.csv",[t_stat.T,p_val.T],delimiter=',')
