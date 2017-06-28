import pandas as pd
import pickle


user_install_path = 'data/user_installedapps.csv'
user_category_path1 = 'saved_file/user_category_1.pkl'
user_category_path2 = 'saved_file/user_category_2.pkl'
user_category_path3 = 'saved_file/user_category_3.pkl'
user_category_path4 = 'saved_file/user_category_4.pkl'

def get_userID_appID():

    userID_appID_1 = pickle.load(open('saved_file/userID_appID_1.pkl','rb'))

    userID_appID_2 = pickle.load(open('saved_file/userID_appID_2.pkl','rb'))

    userID_appID_3 = pickle.load(open('saved_file/userID_appID_3.pkl', 'rb'))

    userID_appID_4 = pickle.load(open('saved_file/userID_appID_4.pkl', 'rb'))

    userID_appID = pd.concat([userID_appID_1,userID_appID_2,userID_appID_3,userID_appID_4],axis=0)

    return userID_appID

def count_user_appCategory(x,use_category_series):

    user_category_list = use_category_series.tolist()

    userID = x['userID'].values[0]
    x_groupby_category = x.groupby('appCategory',as_index = False).count()
    x_category_list = x_groupby_category['appCategory'].tolist()
    data = [userID] + [0] * len(user_category_list)
    for index,category in enumerate(user_category_list):
        if category in x_category_list:
            data[index + 1] = x_groupby_category[x_groupby_category['appCategory']==category]['appID'].values[0]  #count之后,appID那列代表了count数量
    return data

def user_appCategory():

    user_app_category = get_userID_appID()

    user_app_category = user_app_category[:1000][:]
    app_category_series = user_app_category['appCategory'].unique()
    columns_name = ['user_category_count_' + str(i) for i in app_category_series.tolist()]
    user_app_category_groupby_user = user_app_category.groupby('userID',as_index = False)

    user_app_cateogry_list = user_app_category_groupby_user.apply(lambda x:count_user_appCategory(x,app_category_series))
    user_app_cateogry_dataframe = pd.DataFrame(list(user_app_cateogry_list),
                                   columns=['userID'] + columns_name)
    #print(user_app_cateogry_dataframe)
    split_point = len(user_app_cateogry_dataframe) // 4
    data1 = user_app_cateogry_dataframe[:split_point][:]
    data2 = user_app_cateogry_dataframe[split_point:split_point * 2]
    data3 = user_app_cateogry_dataframe[split_point * 2:split_point * 3]
    data4 = user_app_cateogry_dataframe[split_point * 3:][:]

    pickle.dump(data1, open(user_category_path1, 'wb'),protocol=4)
    pickle.dump(data2, open(user_category_path2, 'wb'), protocol=4)
    pickle.dump(data3, open(user_category_path3, 'wb'), protocol=4)
    pickle.dump(data4, open(user_category_path4, 'wb'), protocol=4)

if __name__ == '__main__':
    user_appCategory()