import numpy as np
from sklearn.svm import SVC
import pandas as pd

##added for test and validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

##X samples
#data_full = np.loadtxt('../../../data/MOTIV/test_format_weka.csv',delimiter=',',skiprows=1)
data_full = np.loadtxt('../../../data/MOTIV/augmented_test_data_format_weka.csv',delimiter=',',skiprows=1)
data=data_full[:,0:-1]
#print data

##y values (to predict)
#value = np.loadtxt('class_MM.csv')
value = data_full[:,-1]
df_class= pd.DataFrame(value)

## data preprocessing
from sklearn import preprocessing
#set sparse to False to return dense matrix after transformation and keep all dimensions homogeneous
encod = preprocessing.OneHotEncoder(sparse=False)

feat_to_encode = data[:,0]
#transposition
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.DataFrame(encoded_feature)

feat_to_encode = data[:,1]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,2]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,3]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,4]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,5]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,6]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,7]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,8]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,9]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

#features 10 to 21 are defined between 0 and 1 -> do not touch
#feature 22 is Boolean -> do not touch
for i in range(10,23):
    feat_to_encode = data[:,i]
    feat_to_encode=feat_to_encode.reshape(-1, 1)
    df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(feat_to_encode)],axis=1)

#features 23 to 28 have to put between 0 and 1
encoded_feature = data[:,23]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,24]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,25]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,26]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,27]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,28]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

#features 29 to 34 are defined between 0 and 1 -> do not touch
for i in range(29,35):
    feat_to_encode = data[:,i]
    feat_to_encode=feat_to_encode.reshape(-1, 1)
    df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(feat_to_encode)],axis=1)
## end preprocessing

full_df_binary_encoded = pd.concat([df_binary_encoded,df_class],axis=1)
#full_df_binary_encoded.to_csv("../../../data/MOTIV/test_data_after_preprocessing.csv",sep=',',index=False)
full_df_binary_encoded.to_csv("../../../data/MOTIV/augmented_test_data_after_preprocessing.csv",sep=',',index=False)

##X samples
#data_full = np.loadtxt('../../../data/MOTIV/train_format_weka.csv',delimiter=',',skiprows=1)
data_full = np.loadtxt('../../../data/MOTIV/augmented_train_data_format_weka.csv',delimiter=',',skiprows=1)
data=data_full[:,0:-1]
#print data

##y values (to predict)
#value = np.loadtxt('class_MM.csv')
value = data_full[:,-1]
df_class= pd.DataFrame(value)

## data preprocessing
from sklearn import preprocessing
#set sparse to False to return dense matrix after transformation and keep all dimensions homogeneous
encod = preprocessing.OneHotEncoder(sparse=False)

feat_to_encode = data[:,0]
#transposition
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.DataFrame(encoded_feature)

feat_to_encode = data[:,1]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,2]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,3]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,4]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,5]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,6]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,7]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,8]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

feat_to_encode = data[:,9]
feat_to_encode=feat_to_encode.reshape(-1, 1)
encoded_feature=encod.fit_transform(feat_to_encode)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

#features 10 to 21 are defined between 0 and 1 -> do not touch
#feature 22 is Boolean -> do not touch
for i in range(10,23):
    feat_to_encode = data[:,i]
    feat_to_encode=feat_to_encode.reshape(-1, 1)
    df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(feat_to_encode)],axis=1)

#features 23 to 28 have to put between 0 and 1
encoded_feature = data[:,23]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,24]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,25]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,26]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,27]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

encoded_feature = data[:,28]
ma = np.amax(encoded_feature)
mi = np.amin(encoded_feature)
encoded_feature = (encoded_feature-mi)/(ma-mi)
df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(encoded_feature)],axis=1)

#features 29 to 34 are defined between 0 and 1 -> do not touch
for i in range(29,35):
    feat_to_encode = data[:,i]
    feat_to_encode=feat_to_encode.reshape(-1, 1)
    df_binary_encoded = pd.concat([df_binary_encoded,pd.DataFrame(feat_to_encode)],axis=1)
## end preprocessing

#df_binary_encoded.to_csv("../../../data/MOTIV/train_data_after_preprocessing.csv",sep=',',index=False)
full_df_binary_encoded = pd.concat([df_binary_encoded,df_class],axis=1)
#full_df_binary_encoded.to_csv("../../../data/MOTIV/train_data_after_preprocessing.csv",sep=',',index=False)
full_df_binary_encoded.to_csv("../../../data/MOTIV/augmented_train_data_after_preprocessing.csv",sep=',',index=False)


############# USING SVMs

###using cross valid to select parameters
#
##kernel
clf = SVC(kernel='linear')
#clf = SVC(kernel = 'rbf')

##prepare to split for cross-valid
splitter = KFold(n_splits=5, shuffle=True, random_state = None)
#
##divide data into train and test
x_tr, x_ts, y_tr, y_ts = train_test_split(df_binary_encoded, value, train_size=0.66)
#
##C values to test (kernel='linear')
values = [0.001, 0.01, 1, 10, 100]
#
######## RBF kernel
##C,gamma values to test
#values = [[0.001, 0.1],
#          [0.01, 0.2],
#          [1, 0.5],
#          [10, 2],
#          [100, 0.02],
#          [100, 1]]
#
##retrieve res from different exec
xval_acc_mean = np.zeros((len(values, )))
xval_acc_std = np.zeros((len(values, )))
#
#cross-valid
for i, value_C in enumerate(values):
    #set the C value
    clf.C = value_C
    #create a vector to store accuracy results
    xval_acc = np.zeros((splitter.get_n_splits()))
    k = 0

    print(x_tr.to_numpy())
    #split data and labels into train and test
    for tr_idx, ts_idx in splitter.split(x_tr):
        x_tr_xval = x_tr.to_numpy()[tr_idx]
        y_tr_xval = y_tr[tr_idx]
        x_ts_xval = x_tr.to_numpy()[ts_idx]
        y_ts_xval = y_tr[ts_idx]

        #train a model
        clf.fit(x_tr_xval,y_tr_xval)
        #test the trained model
        yc = clf.predict(x_ts_xval)
        xval_acc[k] = np.mean(yc == y_ts_xval)

        k += 1
    #evaluate accuracy
    xval_acc_mean[i] = xval_acc.mean()
    xval_acc_std[i] = xval_acc.std()
    #print results
    print ('C: ' + str(value_C) + ', avg acc: ' + str(xval_acc_mean[i]) + \
          ' +- ' + str(xval_acc_std[i]))

#retrieve best perf
k = xval_acc_mean.argmax()
best_C = values[k]
clf.C = best_C
#clf.C = 1

####### RBF kernel


##cross-valid
#for i, [value_C, value_gamma] in enumerate(values):
#    clf.C = value_C
#    clf.gamma = value_gamma
#    xval_acc = np.zeros((splitter.get_n_splits()))
#    k=0
#    for tr_idx,ts_idx in splitter.split(x_tr):
#        #split train data for cross-valid
#        x_tr_xval = x_tr[tr_idx,:]
#        y_tr_xval = y_tr[tr_idx]
#        x_ts_xval = x_tr[ts_idx,:]
#        y_ts_xval = y_tr[ts_idx]
#
#        #train a model
#        clf.fit(x_tr_xval,y_tr_xval)
#        #test the trained model
#        yc = clf.predict(x_ts_xval)
#        xval_acc[k] = np.mean(abs(yc - y_ts_xval))
#
#        k += 1
#    #evaluate accuracy
#    xval_acc_mean[i] = xval_acc.mean()
#    xval_acc_std[i] = xval_acc.std()
#    #print results
#    print 'C: ' + str(value_C) + ', avg acc: ' + str(xval_acc_mean[i]) + \
#          ' +- ' + str(xval_acc_std[i])
#
##retrieve best perf
#k = xval_acc_mean.argmax()
#best_C, best_gamma = values[k]
#clf.C = best_C
#clf.gamma = best_gamma
#
#print 'Best C: ' + str(best_C) + ', best gamma: ' + str(best_gamma) + \
#      ", avg acc: " + str(xval_acc_mean[k]) + \
#      ' +- ' + str(xval_acc_std[k])
#
#
######### TEST


##split between test and train already done
clf.fit(x_tr,y_tr)
yc = clf.predict(x_ts)

#print "result: "+str(abs(yc-y_ts))
print (np.mean(yc==y_ts))

###### test attributes regressor
print (len(clf.support_))
print (clf.support_)
print (clf.dual_coef_)
print (clf.coef_)
print (clf.intercept_)

#print 'sizes'
#print clf.support_.shape
#print clf.dual_coef_.shape
#print clf.coef_.shape
#print clf.intercept_.shape


##print 'test'
##res = clf.coef_.T * data[0] + clf.intercept_
##print str(res)
##print res.size

############# SAVE MODEL
from sklearn.externals import joblib
#filename = "../../../data/MOTIV/model_classif_after_preprocessing.txt"
filename = "../../../data/MOTIV/augmented_model_classif_after_preprocessing.txt"
joblib.dump(clf,filename)

##if need to load it again
#clf_loaded = pickle.load(open(filename,'rb'))


