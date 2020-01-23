import pandas as pd

### a script that load data, remove single-valued column and create dummy columns out of the remaining ones
### also, to help for remaining steps, the type of remaining columns are kept into a separated file (with the enumeration of encountered values)
### WARNING: dummy columns are sorted by alphabetical order with respect to encountered values
### save the new matrix in a file
### remaining steps: removing boolean variables (which are now in two columns), removing computation time (sometimes not defined, sometimes not the same format -> seconds or XXmXX)

#load data
df = pd.read_csv("../../../data/JHipster/jhipster_less_column.csv",sep=",",dtype='str')

#get list of column names
header = list(df)

#check type
file_header_type = open("../../../data/JHipster/type_each_col_jhipster_less_column.txt","w")

#remove columns in which only one value is registered
for c in header:
	print(c)

	nb_uniq_value = df[c].unique()
	count = df[c].nunique()
	if count == 1:
		df = df.drop(columns=c)
	elif count == 2:
		file_header_type.write("Bool: "+str(nb_uniq_value)+"\n")
	else:
		file_header_type.write("Enum: "+str(nb_uniq_value)+"\n")

#save original column type
file_header_type.close()

#df_trans = pd.get_dummies(df)
df_trans = pd.DataFrame()

idx_to_dummy=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21]

header_red = list(df)
i=1
for c in header_red:
	if i in idx_to_dummy:
		df_trans = pd.concat([df_trans,pd.get_dummies(df[c])],ignore_index=True,sort=False,axis=1)
	else:
		df_trans = pd.concat([df_trans,df[c]],ignore_index=True,sort=False,axis=1)
	i=i+1


#save reduced df; corresponding to column type
#export_file_path = "./data/jhipster_reduced.csv"
#df.to_csv (export_file_path, index = None, header=True)

export_file_path2 = "../../../data/JHipster/jhipster_transformed.csv"
df_trans.to_csv(export_file_path2, index = None, header=True)
