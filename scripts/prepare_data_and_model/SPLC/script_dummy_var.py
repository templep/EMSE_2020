import numpy as np

import pandas as pd

import pickle


data = np.loadtxt('../../../data/4500_video_config/data_for_ml_reduced_without_classes.csv',delimiter=',')

##try dummy variable
#df = pd.DataFrame(data,columns=[1,2,3,4,5,6])
df = pd.DataFrame(data)

#store index to dummify
index_to_dummify=[0,1,2,3,4,5,6,7,8,9]

#final data frame to concatenate every dimension (included dummified ones)
final_df = pd.DataFrame()

#for each index dummify and store
for idx in index_to_dummify:

	#print idx
	#print df[idx]

	df_dummify = pd.get_dummies(df[idx])
	#concat different dummy variables with other dimensions
	final_df = pd.concat([final_df,df_dummify],axis=1)

	#print final_df

#df_dummify = pd.get_dummies(df[2,3])
#add remaining columns
#final_df = pd.concat([final_df,df[:;10:]],axis=1)
tmp = df.iloc[:,10:]
final_df = pd.concat([final_df,tmp],axis=1)

final_df.to_csv('test.csv',index=False)
