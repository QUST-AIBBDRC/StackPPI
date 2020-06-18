import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model.logistic import LogisticRegression
import utils.tools as utils
import os
import sys
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]

yeast_data=sio.loadmat(file1)
#yeast_data=sio.loadmat('DNN_yeast_six.mat')
protein=yeast_data.get('feature')
feature=np.array(protein)
label_P=np.ones((int(feature.shape[0]/2),1))
label_N=np.zeros((int(feature.shape[0]/2),1))
label=np.append(label_P,label_N)
shu_=np.column_stack([label,feature])
[row1,column1]=np.shape(shu_)
index = [i for i in range(row1)]
np.random.shuffle(index)#shuffle the index
index=np.array(index)
data_=shu_[index,:]
shu=data_[:,np.array(range(1,column1))]
label=data_[:,0]
in_data=sio.loadmat(file2)
in_shu=in_data.get('feature')
in_label=np.ones((in_shu.shape[0],1))

scaler = preprocessing.StandardScaler().fit(shu)
train_data=scaler.transform(shu) 
test_data=scaler.transform(in_shu)

fuhao=sio.loadmat(file3)
mask=fuhao.get('mask')
train_shu=train_data[:,mask]
train_shu=np.reshape(train_shu,(train_shu.shape[0],train_shu.shape[2]))
test_shu=test_data[:,mask]
test_shu=np.reshape(test_shu,(test_shu.shape[0],test_shu.shape[2]))

train_label=label
test_label=in_label

num_class=2
def get_stacking(clf, x_train, y_train, x_test, num_class,n_folds=5):

    kf = KFold(n_splits=n_folds)
    second_level_train_set=[]
    test_nfolds_set=[]
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        second_level_train_ = clf.predict_proba(x_tst)
        second_level_train_set.append(second_level_train_)
        test_nfolds= clf.predict_proba(x_test)
        test_nfolds_set.append(test_nfolds)   
    train_second=second_level_train_set
    train_second_level=np.concatenate((train_second[0],train_second[1],train_second[2],train_second[3],train_second[4]),axis=0) 
    test_second_level_=np.array(test_nfolds_set)   
    test_second_level=np.mean(test_second_level_,axis = 0)
    return train_second_level,test_second_level
def get_first_level(train_x, train_y, test_x,num_class):
    #lgm_model1 = lgb.LGBMClassifier(n_estimators=500,num_leaves=256)
    #xbt_model1 = xgb.XGBClassifier(max_depth=15, n_estimators=500, objective="binary:logistic",n_jobs=-1)
    rf_model1=RandomForestClassifier(max_depth=10,n_estimators=500)
    ext_model1=ExtraTreesClassifier(n_estimators=500)
    rf_model2=RandomForestClassifier(max_depth=10,n_estimators=500)
    ext_model2=ExtraTreesClassifier(n_estimators=500)
#    gtb_model1= GradientBoostingClassifier(max_depth=10, n_estimators=500)
#    et_model1= ExtraTreesClassifier(max_depth=10, n_estimators=500,n_jobs=-1)
    train_sets = []
    test_sets = []
    for clf in [rf_model1,ext_model1,rf_model2,ext_model2]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x,num_class)
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train = np.concatenate([result_set.reshape(-1,num_class) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,num_class) for y_test_set in test_sets], axis=1)
    return meta_train,meta_test
meta_train,meta_test=get_first_level(train_data,train_label,test_data,num_class)

LR=LogisticRegression()
LR.fit(meta_train, train_label)
pre_score = LR.predict_proba(meta_test)
pre_class= utils.categorical_probas_to_classes(pre_score)
test_class=test_label
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(pre_class), pre_class, test_class)
print("Testing Accuracy Hsapi:",acc)

