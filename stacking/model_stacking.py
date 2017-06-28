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

#dataframe转为array
train_data = pd.DataFrame.as_matrix(train_data)
test_data = pd.DataFrame.as_matrix(test_data)
valid_data = pd.DataFrame.as_matrix(valid_data)

train_label = pd.DataFrame.as_matrix(train_label)
valid_label = pd.DataFrame.as_matrix(valid_label)

#xgboost参数
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'              #多分类：'multi:softprob'
param['eval_metric '] = 'logloss' #校验数据所需要的评价指标
param['eta'] = 0.1  #通常最后设置eta为0.01~0.2
# param['min_child_weight']=0.5 #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
# param['alpha '] =1 #默认0，L1正则惩罚系数，当数据维度极高时可以使用，使得算法运行更快。
# param['lambda '] =1 #默认0，L2 正则的惩罚系数
# param['scale_pos_weight'] = 0.025 #默认0，大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
param['max_depth'] = 6  #通常取值：3-10
#     param['colsample_bytree '] =1 #默认为1，在建立树时对特征随机采样的比例。
# param['subsample']=0.8 #默认为1，用于训练模型的子样本占整个样本集合的比例。
# param['max_delta_step']=0.3  #通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果
param['silent'] = 1  #取1时表示打印出运行时信息，取0时表示以缄默方式运行，不打印运行时的信息。
#     param['nthread'] = 4  #如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
#     param['num_class'] = 2  #多分类时需设置
num_round = 300  #提升迭代的个数

folds_num = 5  # 交叉验证的次数
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

            dataset_blend_test_j[:, j] = bst.predict( xgb_valid_X )  # 对验证集预测 n_folds 次
            dataset_blend_predict_j[:, j] = bst.predict( xgb_test_data )
        else:
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]  # 取出正类概率
            dataset_blend_train[test_index] = y_submission
            temp_prob = clf.predict_proba(valid_data)[:, 1]
            dataset_blend_test_j[:, j] = temp_prob  # 对验证集预测 n_folds 次
            print(compute_loss_and_save.logloss(valid_label,temp_prob))
            dataset_blend_predict_j[:, j] = clf.predict_proba(test_data)[:, 1]
        j += 1
    dataset_blend_test = dataset_blend_test_j.mean(1)  # 取6折的均值作为最后测试特征
    dataset_blend_predict = dataset_blend_predict_j.mean(1)

    ###保存结果
    pickle.dump(dataset_blend_train,open('feature_xwd/train_feat_'+name+'.pkl', 'wb'),protocol=4)
    pickle.dump(dataset_blend_test,open('feature_xwd/valid_feat_'+name+'.pkl', 'wb'),protocol=4)
    pickle.dump(dataset_blend_predict,open('feature_xwd/test_feat_'+name+'.pkl', 'wb'),protocol=4)

'''
print("Blending result：")
clf = LogisticRegression()
clf.fit(dataset_blend_train, train_label)
pred_prob = clf.predict_proba(dataset_blend_test)
loss = compute_loss_and_save.logloss(valid_label,pred_prob)
print('验证集：',loss)
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
