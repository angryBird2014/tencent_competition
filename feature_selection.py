from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import pickle
import pandas as pd

train_data = pickle.load(open('xwd/train_feat.pkl','rb'))
train_label = pickle.load(open('xwd/train_label.pkl','rb'))
'''
train_data = train_data[:100]
train_label =  train_label[:100]
train_data.fillna(0,inplace=True)
train_label.fillna(0,inplace=True)
'''
valid_data = pickle.load(open('xwd/valid_feat.pkl','rb'))
valid_label = pickle.load(open('xwd/valid_label.pkl','rb'))
test = pickle.load(open('xwd/test_feat.pkl','rb'))

gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=200, #600W
                        learning_rate=0.05,
                        min_child_samples=100,
                        n_estimators=1)



model = RFE(estimator=LogisticRegression(), n_features_to_select=70).fit(train_data,train_label)

proba_test = model.predict_proba(test)




