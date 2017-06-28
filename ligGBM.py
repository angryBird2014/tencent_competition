

import lightgbm as lgb
import pickle
import compute_loss_and_save
import pandas as pd


def delFeature(data):
    #exlude ,'user_app_rate','app_user_rate'
    columns_name = ['clickTime','user_app_rate','app_user_rate','user_appInAllApp_num','user_appInAllApp_rate','user_DiffAdvertiser_num','user_DiffAdvertiser_rate',
                    'user_category_count_407','user_category_count_210','user_category_count_2','user_category_count_205','user_category_count_204',
                    'user_category_count_401','user_category_count_103','user_category_count_101','user_category_count_211','user_category_count_110',
                    'user_category_count_107','user_category_count_105','user_category_count_408','user_category_count_102','user_category_count_409',
                    'user_category_count_1','user_category_count_109','user_category_count_406','user_category_count_303','user_category_count_403',
                    'pos_user_conver_amount','user_con_conver_amount','userID_sitesetID_conver_amount','userAmount_combine_appID_groupby_count','0_6_user_conver_amount',
                    '0_6_user_conver_rate','6_12_user_conver_amount','6_12_user_conver_rate','12_18_user_conver_amount','12_18_user_conver_rate','18_24_user_conver_amount']
    data = data.drop(columns_name,axis = 1)
    return data


# load or create your dataset
print('Load data...')
train_data = pickle.load(open('data/train_feat.pkl','rb'))
user_active_before28 = pickle.load(open('saved_file/user_active_amount_before28.pkl','rb'))
creativeID_active_before28 = pickle.load(open('saved_file/creativeID_active_amount_before28.pkl','rb'))
train_data = pd.merge(train_data,user_active_before28,how='left',on='userID')
train_data = pd.merge(train_data,creativeID_active_before28,how='left',on='creativeID')


train_label = pickle.load(open('data/train_label.pkl','rb'))

valid_data = pickle.load(open('data/valid_feat.pkl','rb'))
valid_data = pd.merge(valid_data,user_active_before28,how='left',on='userID')
valid_data = pd.merge(valid_data,creativeID_active_before28,how='left',on='creativeID')
del user_active_before28
del creativeID_active_before28

valid_label = pickle.load(open('data/valid_label.pkl','rb'))


test = pickle.load(open('data/test_feat.pkl','rb'))
user_active_before31 = pickle.load(open('saved_file/user_active_amount_before31.pkl','rb'))
creativeID_active_before31 = pickle.load(open('saved_file/creativeID_active_amount_before31.pkl','rb'))
test = pd.merge(test,user_active_before31,how='left',on='userID')
test = pd.merge(test,creativeID_active_before31,how='left',on='creativeID')
del user_active_before31
del creativeID_active_before31

train_data = delFeature(train_data)
valid_data = delFeature(valid_data)
test = delFeature(test)
'''
pickle.dump(train_data,open('xwd/train_feat.pkl','wb'),protocol=4)
pickle.dump(train_label,open('xwd/train_label.pkl','wb'),protocol=4)
pickle.dump(valid_data,open('xwd/valid_feat.pkl','wb'),protocol=4)
pickle.dump(valid_label,open('xwd/valid_label.pkl','wb'),protocol=4)
pickle.dump(test,open('xwd/test_feat.pkl','wb'),protocol=4)
'''

print('Start training...')
# train

gbm = lgb.LGBMClassifier(objective='binary',max_bin=126,
                        num_leaves=256, #600W
                        learning_rate=0.05,
                        n_estimators=20000)
gbm.fit(train_data, train_label,
        eval_set=[(valid_data, valid_label)],
        eval_metric='logloss',
        early_stopping_rounds=10)


print('Calculate feature importances...')
# feature importances
feature_importance = list(gbm.feature_importances_)
columns = train_data.columns.tolist()
dist = {}
for k,v in zip(columns,feature_importance):
    dist[k]=v
    print(k,':',v)
pickle.dump(dist,open('saved_file/feature_importance','wb'))
print('Start predicting...')

# predict
y_pred = gbm.predict_proba(valid_data, num_iteration=gbm.best_iteration)
# eval
y_pred = y_pred[:,1]
loss = compute_loss_and_save.logloss(valid_label,y_pred)
print('The rmse of prediction is:', loss)

proba_test = gbm.predict_proba(test,num_iteration=gbm.best_iteration)
proba_test = proba_test[:,1]
compute_loss_and_save.submission(proba_test)

