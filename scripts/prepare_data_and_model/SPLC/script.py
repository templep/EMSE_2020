import numpy as np
from sklearn.svm import SVC

##added for test and validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

##X samples
data_full = np.loadtxt('../../../data/MOTIV/train_format_weka.csv',delimiter=',',skiprows=1)
data=data_full[:,0:-1]
#print data

##y values (to predict)
#value = np.loadtxt('class_MM.csv')
value = data_full[:,-1]


##stats on data
##print 'mean value of y: '+str(value.mean())
##print 'stdev value of y: '+str(value.std())


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
x_tr, x_ts, y_tr, y_ts = train_test_split(data,value, train_size=0.66)
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
    #split data and labels into train and test
    for tr_idx, ts_idx in splitter.split(x_tr):
        x_tr_xval = x_tr[tr_idx,:]
        y_tr_xval = y_tr[tr_idx]
        x_ts_xval = x_tr[ts_idx,:]
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
filename = "../../../data/MOTIV/model_classif.txt"
joblib.dump(clf,filename)

##if need to load it again
#clf_loaded = pickle.load(open(filename,'rb'))


