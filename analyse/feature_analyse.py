import pandas as pd
import pickle
import numpy as np
if __name__ == '__main__':
    train_file = '../saved_file/merge_src_train_data.pkl'
    train_data = pickle.load(open(train_file,'rb'))
    '''
    user_app_rate   app_user_rate
    '''
    '''
    feature_names = ['user_app_rate','app_user_rate']
    for feature_name in feature_names:
        print('*******',feature_name,'********')
        pos_average = np.mean(train[train['label'] == 1][feature_name])
        pos = np.var(train[train['label'] == 1][feature_name])

        neg_average = np.mean(train[train['label'] == 0][feature_name])
        neg = np.var(train[train['label'] == 0][feature_name])

        print(pos_average, neg_average)
        print(pos, neg)
    '''
    train_data['userAmount_combine_appID_groupby_count'] = train_data['userAmount'] + train_data['appID_groupby_count']


    feature_names = ['userAmount_combine_appID_groupby_count']
    for feature_name in feature_names:
        print('*******', feature_name, '********')
        pos_average = np.mean(train_data[train_data['label'] == 1][feature_name])
        pos = np.var(train_data[train_data['label'] == 1][feature_name])

        neg_average = np.mean(train_data[train_data['label'] == 0][feature_name])
        neg = np.var(train_data[train_data['label'] == 0][feature_name])

        print(pos_average, neg_average)
        print(pos, neg)

