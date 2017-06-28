# coding=gbk
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier, Perceptron
import compute_loss_and_save
import xgboost as xgb
import pickle
import scipy as sp


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

# submission
def submission(pred):
    dfTest = pd.read_csv("../data/test.csv")
    print(len(dfTest['instanceID']))
    print(len(pred))
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": pred})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)


names = ['RF','xgb','RF_entropy','lightgbm']
train_label = pickle.load(open('../feature_xwd/train_label.pkl','rb'))
valid_label = pickle.load(open('../feature_xwd/valid_label.pkl','rb'))
train_feat = np.zeros((len(train_label), len(names)))
valid_feat = np.zeros((len(valid_label), len(names)))
test_feat = np.zeros((3321748, len(names)))

j = 0
for name in names:
    train_feat_j = pickle.load(open('train_feat_'+name+'.pkl','rb'))
    valid_feat_j = pickle.load(open('valid_feat_'+name+'.pkl','rb'))
    test_feat_j = pickle.load(open('test_feat_'+name+'.pkl','rb'))

    train_feat[:,j] = train_feat_j
    valid_feat[:,j] = valid_feat_j
    test_feat[:,j] = test_feat_j

    j += 1

#xgboost����
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'              #����ࣺ'multi:softprob'
param['eval_metric '] = 'logloss' #У����������Ҫ������ָ��
param['eta'] = 0.001  #ͨ���������etaΪ0.01~0.2
# param['min_child_weight']=0.5 #���ӽڵ�����С������Ȩ�غ͡����һ��Ҷ�ӽڵ������Ȩ�غ�С��min_child_weight���ֹ��̽�����
# param['alpha '] =1 #Ĭ��0��L1����ͷ�ϵ����������ά�ȼ���ʱ����ʹ�ã�ʹ���㷨���и��졣
# param['lambda '] =1 #Ĭ��0��L2 ����ĳͷ�ϵ��
# param['scale_pos_weight'] = 0.025 #Ĭ��0������0��ȡֵ���Դ������ƽ������������ģ�͸�������
param['max_depth'] = 3  #ͨ��ȡֵ��3-10
#     param['colsample_bytree '] =1 #Ĭ��Ϊ1���ڽ�����ʱ��������������ı�����
# param['subsample']=0.8 #Ĭ��Ϊ1������ѵ��ģ�͵�������ռ�����������ϵı�����
# param['max_delta_step']=0.3  #ͨ������Ҫ�������ֵ������ʹ��logistics �ع�ʱ������𼫶Ȳ�ƽ�⣬������ò���������Ч��
param['silent'] = 1  #ȡ1ʱ��ʾ��ӡ������ʱ��Ϣ��ȡ0ʱ��ʾ�Լ�Ĭ��ʽ���У�����ӡ����ʱ����Ϣ��
#     param['nthread'] = 4  #�����ϣ��������ٶ����У����鲻�������������ģ�ͽ��Զ��������߳�
#     param['num_class'] = 2  #�����ʱ������
num_round = 300  #���������ĸ���

print("Blending result��")

clf = LogisticRegression()
clf.fit(train_feat, train_label)
pred_prob = clf.predict_proba(valid_feat)[:, 1]
loss = logloss(valid_label,pred_prob)
print('��֤����',loss)
predict_prob = clf.predict_proba(test_feat)[:, 1]
submission(predict_prob)

'''
xg_train = xgb.DMatrix( train_feat, label=train_label)
xg_test = xgb.DMatrix(valid_feat, label=valid_label)
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
bst = xgb.train(param, xg_train, num_round,evals=watchlist,early_stopping_rounds=20)  #, watchlist,early_stopping_rounds=10
y_submission = bst.predict( xg_test )
loss = logloss(valid_label,y_submission)
print("loss",loss)
xgb_test_data = xgb.DMatrix(test_feat)
predict_prob = bst.predict( xgb_test_data )  # ����֤��Ԥ�� n_folds ��
print(predict_prob[0:10])
submission(predict_prob)
'''


