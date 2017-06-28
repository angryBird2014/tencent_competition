import pandas as pd
import numpy  as np
import pickle



split_day = 28
adID_age_train_path = 'saved_file/CVR_age_adID_before28.pkl'
adID_age_test_path = 'saved_file/CVR_age_adID_before31.pkl'

appID_age_train_path = 'saved_file/CVR_age_appID_before28.pkl'
appID_age_test_path = 'saved_file/CVR_age_appID_before31.pkl'

age_appPlatform_train_path = 'saved_file/CVR_age_appPlatform_before28.pkl'
age_appPlatform_test_path = 'saved_file/CVR_age_appPlatform_before31.pkl'

creativeID_age_train_path = 'saved_file/CVR_creativeID_age_before28.pkl'
creativeID_age_test_path = 'saved_file/CVR_creativeID_age_before31.pkl'

#age_adID
def age_adID_CVR_rate(data):
    age = data['age'].values[0]
    adID = data['adID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([adID,age,click_amount,conversation_amount,rate])

def get_adID_age_train():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['adID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_adID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['adID','age','adID_age_click_amount', 'adID_age_conver_amount','CVR_adID_age'])
    pickle.dump(data,open(adID_age_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_adID_age_test():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))



    train_position_connect = train.groupby(['adID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_adID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['adID','age','adID_age_click_amount', 'adID_age_conver_amount','CVR_adID_age'])
    pickle.dump(data,open(adID_age_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list

#age_appID
def age_appID_CVR_rate(data):
    age = data['age'].values[0]
    appID = data['appID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([appID,age,click_amount,conversation_amount,rate])

def get_appID_age_train():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['appID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_appID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['appID','age','appID_age_click_amount', 'appID_age_conver_amount','CVR_appID_age'])
    pickle.dump(data,open(appID_age_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_appID_age_test():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))



    train_position_connect = train.groupby(['appID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_appID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['appID','age','appID_age_click_amount', 'appID_age_conver_amount','CVR_appID_age'])
    pickle.dump(data,open(appID_age_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list


#age_appPlatform
def age_appPlatform_CVR_rate(data):
    age = data['age'].values[0]
    appPlatform = data['appPlatform'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([age,appPlatform,click_amount,conversation_amount,rate])

def get_age_appPlatform_train():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['age','appPlatform'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_appPlatform_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age','appPlatform','age_appPlatform_click_amount', 'age_appPlatform_conver_amount','CVR_age_appPlatform'])
    pickle.dump(data,open(age_appPlatform_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_appPlatform_test():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))



    train_position_connect = train.groupby(['age','appPlatform'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_appPlatform_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age','appPlatform','age_appPlatform_click_amount', 'age_appPlatform_conver_amount','CVR_age_appPlatform'])
    pickle.dump(data,open(age_appPlatform_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list


#age_creativeID
def age_creativeID_CVR_rate(data):
    age = data['age'].values[0]
    creativeID = data['creativeID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([creativeID,age,click_amount,conversation_amount,rate])

def get_age_creativeID_train():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['creativeID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_creativeID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['creativeID','age','creativeID_age_click_amount', 'creativeID_age_conver_amount','CVR_creativeID_age'])
    pickle.dump(data,open(creativeID_age_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_creativeID_test():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))



    train_position_connect = train.groupby(['creativeID','age'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_creativeID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['creativeID_age','age','creativeID_age_click_amount', 'creativeID_age_conver_amount','CVR_creativeID_age'])
    pickle.dump(data,open(creativeID_age_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list


if __name__ == '__main__':

    get_adID_age_train()
    get_adID_age_test()
    get_appID_age_train()
    get_appID_age_test()
    get_age_appPlatform_train()
    get_age_appPlatform_test()
    get_age_creativeID_train()
    get_age_creativeID_test()
