import scipy.io as sio
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier,ExtraTreesClassifier)
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import scale
import utils.tools as utils
from sklearn.metrics import roc_curve, auc
from dimension_reduction import KPCA,LLE,TSVD,mds
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
file1 = sys.argv[1]
yeast_data=sio.loadmat(file1)
#yeast_data=sio.loadmat('DNN_yeast_six.mat')
protein=yeast_data.get('feature')
row1=protein.shape[0]
feature=np.array(protein)
label_P=np.ones((int(row1/2),1))
label_N=np.zeros((int(row1/2),1))
label=np.append(label_P,label_N)
#label=np.concatenate((label_P,label_N),axis=0)
train_data1=feature
protein_label=label.T.ravel()
protein_label=protein_label.astype(float)
train_label1=protein_label


shu_1=np.column_stack([train_label1,train_data1])
(row,column)=shu_1.shape
shu=shu_1[:,np.array(range(1,column))]
label=shu_1[:,0]

shu_scale=scale(shu)
train_shu=LLE(shu_scale,n_components=300)
index = [i for i in range(row)]
np.random.shuffle(index)#shuffle the index
index=np.array(index)
train_data=train_shu[index,:]
train_label=label[index]


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
    rf_model1=RandomForestClassifier(max_depth=10,n_estimators=500)
    ext_model1=ExtraTreesClassifier(n_estimators=500)
    rf_model2=RandomForestClassifier(max_depth=10,n_estimators=500)
    ext_model2=ExtraTreesClassifier(n_estimators=500)
    train_sets = []
    test_sets = []
    for clf in [rf_model1,ext_model1,rf_model2,ext_model2]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x,num_class)
        train_sets.append(train_set)
        test_sets.append(test_set)
    meta_train = np.concatenate([result_set.reshape(-1,num_class) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1,num_class) for y_test_set in test_sets], axis=1)
    return meta_train,meta_test
def get_second_level(train_dim,train_label,test_dim,num_class):
    meta_train,meta_test=get_first_level(train_dim,train_label,test_dim,num_class)
    LR=LogisticRegression()
    hist=LR.fit(meta_train,train_label)
    pre_score=LR.predict_proba(meta_test)
    return meta_train,meta_test,pre_score 
skf= StratifiedKFold(n_splits=5)
sepscores = []
num_class=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
for train, test in skf.split(train_data,train_label): 
    meta_train,meta_test,y_score=get_second_level(train_data[train], train_label[train],train_data[test],num_class)
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(train_label[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=train_label[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('RF:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))


result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
#column=data.shape[1]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('M_LLE_yscore.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('M_LLE_ytest.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('M_LLE_result.csv')
