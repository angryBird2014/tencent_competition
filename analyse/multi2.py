import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import time
import pickle
from functools import partial
import multiprocessing as mp

user_category_path1 = '../saved_file/user_category_1.pkl'
user_category_path2 = '../saved_file/user_category_2.pkl'
user_category_path3 = '../saved_file/user_category_3.pkl'
user_category_path4 = '../saved_file/user_category_4.pkl'




def papply(groups, fxn,fxn_parameter):
    ''' apply one process to each df group in groups
    '''

    func = partial(fxn,fxn_parameter)
    result = Parallel(n_jobs=mp.cpu_count() - 6)(delayed(func)(group) for name, group in groups)


    return result


def count_user_appCategory(user_category_list,x): #x tuple

    user_category_list = user_category_list.tolist()
    userID = x['userID'].values[0]
    data = [userID] + [0] * len(user_category_list)
    for index, category in enumerate(user_category_list):
        data[index + 1] = len(x[x['appCategory'] == category])  #
    return data

def user_appCategory():

    user_app_category = get_userID_appID()
    app_category = pd.read_csv('../data/app_categories.csv')


    app_category_series = app_category['appCategory'].unique()
    columns_name = ['user_category_count_' + str(i) for i in app_category_series.tolist()]

    user_app_category_groupby_user = user_app_category.groupby('userID',as_index = False)


    user_app_cateogry_list = papply(user_app_category_groupby_user,count_user_appCategory,app_category_series)


    user_app_cateogry_dataframe = pd.DataFrame(list(user_app_cateogry_list),
                                   columns=['userID'] + columns_name)
    #print(user_app_cateogry_dataframe)
    user_app_cateogry_dataframe.drop_duplicates(inplace=True)
    pickle.dump(user_app_cateogry_dataframe,open('../saved_file/user_category_before31.pkl','wb'),protocol=4)
    '''
    split_point = len(user_app_cateogry_dataframe) // 4
    data1 = user_app_cateogry_dataframe[:split_point][:]
    data2 = user_app_cateogry_dataframe[split_point:split_point * 2]
    data3 = user_app_cateogry_dataframe[split_point * 2:split_point * 3]
    data4 = user_app_cateogry_dataframe[split_point * 3:][:]

    pickle.dump(data1, open(user_category_path1, 'wb'),protocol=4)
    pickle.dump(data2, open(user_category_path2, 'wb'), protocol=4)
    pickle.dump(data3, open(user_category_path3, 'wb'), protocol=4)
    pickle.dump(data4, open(user_category_path4, 'wb'), protocol=4)
    '''

def get_userID_appID():

    '''
    userID_appID_1 = pickle.load(open('../saved_file/userID_appID_1.pkl','rb'))

    userID_appID_2 = pickle.load(open('../saved_file/userID_appID_2.pkl','rb'))

    userID_appID_3 = pickle.load(open('../saved_file/userID_appID_3.pkl', 'rb'))

    userID_appID_4 = pickle.load(open('../saved_file/userID_appID_4.pkl', 'rb'))

    userID_appID = pd.concat([userID_appID_1,userID_appID_2,userID_appID_3,userID_appID_4],axis=0)
    '''
    userID_appID = pickle.load(open('../saved_file/userID_appID_before31.pkl','rb'))
    return userID_appID

if __name__ == '__main__':
    user_appCategory()