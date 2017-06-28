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

#xgboost参数
param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'              #多分类：'multi:softprob'
param['eval_metric '] = 'logloss' #校验数据所需要的评价指标
param['eta'] = 0.001  #通常最后设置eta为0.01~0.2
# param['min_child_weight']=0.5 #孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
# param['alpha '] =1 #默认0，L1正则惩罚系数，当数据维度极高时可以使用，使得算法运行更快。
# param['lambda '] =1 #默认0，L2 正则的惩罚系数
# param['scale_pos_weight'] = 0.025 #默认0，大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
param['max_depth'] = 3  #通常取值：3-10
#     param['colsample_bytree '] =1 #默认为1，在建立树时对特征随机采样的比例。
# param['subsample']=0.8 #默认为1，用于训练模型的子样本占整个样本集合的比例。
# param['max_delta_step']=0.3  #通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果
param['silent'] = 1  #取1时表示打印出运行时信息，取0时表示以缄默方式运行，不打印运行时的信息。
#     param['nthread'] = 4  #如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
#     param['num_class'] = 2  #多分类时需设置
num_round = 300  #提升迭代的个数

print("Blending result：")

clf = LogisticRegression()
clf.fit(train_feat, train_label)
pred_prob = clf.predict_proba(valid_feat)[:, 1]
loss = logloss(valid_label,pred_prob)
print('验证集：',loss)
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
predict_prob = bst.predict( xgb_test_data )  # 对验证集预测 n_folds 次
print(predict_prob[0:10])
submission(predict_prob)
'''


