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

train_data = pickle.load(open('feature_set/train_feat.pkl','rb'))
test_data = pickle.load(open('feature_set/test_feat.pkl', 'rb'))
train_label = pickle.load(open('feature_set/train_label.pkl', 'rb'))

valid_data = pickle.load(open('feature_set/valid_feat.pkl', 'rb'))
valid_label = pickle.load(open('feature_set/valid_label.pkl', 'rb'))
train_data = train_data.fillna(0)
valid_data = valid_data.fillna(0)
test_data = test_data.fillna(0)

#dataframeתΪarray
train_data = pd.DataFrame.as_matrix(train_data)
test_data = pd.DataFrame.as_matrix(test_data)
valid_data = pd.DataFrame.as_matrix(valid_data)

train_label = pd.DataFrame.as_matrix(train_label)
valid_label = pd.DataFrame.as_matrix(valid_label)

#xgboost����
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'              #����ࣺ'multi:softprob'
param['eval_metric '] = 'logloss' #У����������Ҫ������ָ��
param['eta'] = 0.1  #ͨ���������etaΪ0.01~0.2
# param['min_child_weight']=0.5 #���ӽڵ�����С������Ȩ�غ͡����һ��Ҷ�ӽڵ������Ȩ�غ�С��min_child_weight���ֹ��̽�����
# param['alpha '] =1 #Ĭ��0��L1����ͷ�ϵ����������ά�ȼ���ʱ����ʹ�ã�ʹ���㷨���и��졣
# param['lambda '] =1 #Ĭ��0��L2 ����ĳͷ�ϵ��
# param['scale_pos_weight'] = 0.025 #Ĭ��0������0��ȡֵ���Դ������ƽ������������ģ�͸�������
param['max_depth'] = 6  #ͨ��ȡֵ��3-10
#     param['colsample_bytree '] =1 #Ĭ��Ϊ1���ڽ�����ʱ��������������ı�����
# param['subsample']=0.8 #Ĭ��Ϊ1������ѵ��ģ�͵�������ռ�����������ϵı�����
# param['max_delta_step']=0.3  #ͨ������Ҫ�������ֵ������ʹ��logistics �ع�ʱ������𼫶Ȳ�ƽ�⣬������ò���������Ч��
param['silent'] = 1  #ȡ1ʱ��ʾ��ӡ������ʱ��Ϣ��ȡ0ʱ��ʾ�Լ�Ĭ��ʽ���У�����ӡ����ʱ����Ϣ��
#     param['nthread'] = 4  #�����ϣ��������ٶ����У����鲻�������������ģ�ͽ��Զ��������߳�
#     param['num_class'] = 2  #�����ʱ������
num_round = 300  #���������ĸ���

folds_num = 5  # ������֤�Ĵ���
'''
clfs = [ExtraTreesClassifier(n_estimators=350, n_jobs=-1, criterion='gini'),#300
        ExtraTreesClassifier(n_estimators=350, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.1, subsample=0.5, max_depth=6, n_estimators=350), #350
        RandomForestClassifier(n_estimators=350, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=350, n_jobs=-1, criterion='entropy'),
        xgb]
'''
def classify(clf,name):
    j = 0
    for train_index, test_index in zip(cv_train_index, cv_test_index):

        X_train = train_data[train_index]
        y_train = train_label[train_index]
        X_test = train_data[test_index]
        y_test = train_label[test_index]
        if name=='xgb':
            xg_train = xgb.DMatrix( X_train, label=y_train)
            xg_test = xgb.DMatrix(X_test, label=y_test)
            # watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
            bst = xgb.train(param, xg_train, num_round)  #, watchlist,early_stopping_rounds=10
            y_submission = bst.predict( xg_test )
            dataset_blend_train[test_index] = y_submission
            xgb_valid_X = xgb.DMatrix(valid_data)
            xgb_test_data = xgb.DMatrix(test_data)

            dataset_blend_test_j[:, j] = bst.predict( xgb_valid_X )  # ����֤��Ԥ�� n_folds ��
            dataset_blend_predict_j[:, j] = bst.predict( xgb_test_data )
        else:
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]  # ȡ���������
            dataset_blend_train[test_index] = y_submission
            temp_prob = clf.predict_proba(valid_data)[:, 1]
            dataset_blend_test_j[:, j] = temp_prob  # ����֤��Ԥ�� n_folds ��
            print(compute_loss_and_save.logloss(valid_label,temp_prob))
            dataset_blend_predict_j[:, j] = clf.predict_proba(test_data)[:, 1]
        j += 1
    dataset_blend_test = dataset_blend_test_j.mean(1)  # ȡ6�۵ľ�ֵ��Ϊ����������
    dataset_blend_predict = dataset_blend_predict_j.mean(1)

    ###������
    pickle.dump(dataset_blend_train,open('feature_xwd/train_feat_'+name+'.pkl', 'wb'),protocol=4)
    pickle.dump(dataset_blend_test,open('feature_xwd/valid_feat_'+name+'.pkl', 'wb'),protocol=4)
    pickle.dump(dataset_blend_predict,open('feature_xwd/test_feat_'+name+'.pkl', 'wb'),protocol=4)

'''
print("Blending result��")
clf = LogisticRegression()
clf.fit(dataset_blend_train, train_label)
pred_prob = clf.predict_proba(dataset_blend_test)
loss = compute_loss_and_save.logloss(valid_label,pred_prob)
print('��֤����',loss)
predict_prob = clf.predict_proba(dataset_blend_predict)
compute_loss_and_save.submission(predict_prob)
'''
if __name__ == '__main__':
    dataset_blend_train = np.zeros(train_label.shape[0])
    dataset_blend_test = np.zeros(valid_label.shape[0])
    dataset_blend_predict = np.zeros(len(test_data))
    dataset_blend_test = np.zeros(len(valid_data))
    dataset_blend_test_j = np.zeros((len(valid_label), folds_num))
    dataset_blend_predict_j = np.zeros((len(test_data), folds_num))

    cv_train_index = pickle.load(open('feature_xwd/cv_train_index.pkl','rb'))
    cv_test_index = pickle.load(open('feature_xwd/cv_test_index.pkl','rb'))

    print('train begin!')
    clf = RandomForestClassifier(n_estimators=800, n_jobs=-1, criterion='gini')
    classify(clf,'RF')
