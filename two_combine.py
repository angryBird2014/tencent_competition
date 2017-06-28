import pandas as pd
import numpy  as np
import pickle
import gc
from functools import partial
import multiprocessing as mp


split_day = 28

pos_residence_train_path = 'saved_file/CVR_pos_residence_province_before28.pkl'
pos_residence_test_path = 'saved_file/CVR_pos_residence_province_before31.pkl'

age_hometown_train_path = 'saved_file/CVR_age_hometown_province_before28.pkl'
age_hometown_test_path = 'saved_file/CVR_age_hometown_province_before31.pkl'

age_residence_train_path = 'saved_file/CVR_age_residence_province_before28.pkl'
age_residence_test_path = 'saved_file/CVR_age_residence_province_before31.pkl'

userID_sitesetID_train_path = 'saved_file/CVR_userID_sitesetID_before28.pkl'
userID_sitesetID_test_path = 'saved_file/CVR_userID_sitesetID_before31.pkl'

camgaign_connectionType_train_path = 'saved_file/CVR_camgaign_connectionType_before28.pkl'
camgaign_connectionType_test_path = 'saved_file/CVR_camgaign_connectionType_before31.pkl'

userID_appPlatform_train_path = 'saved_file/CVR_userID_appPlatform_before28.pkl'
userID_appPlatform_test_path = 'saved_file/CVR_userID_appPlatform_before31.pkl'


userID_clickDay_train_path = 'saved_file/CVR_userID_clickDay_before28.pkl'
userID_clickDay_test_path = 'saved_file/CVR_userID_clickDay_before31.pkl'

age_hot_camgaignID_train_path = 'saved_file/CVR_age_hot_camgaignID_before28.pkl'
age_hot_camgaignID_test_path = 'saved_file/CVR_age_hot_camgaignID_before31.pkl'

positionID_appCategoryFirstClass_train_path = 'saved_file/CVR_positionID_appCategoryFirstClass_before28.pkl'
positionID_appCategoryFirstClass_test_path = 'saved_file/CVR_positionID_appCategoryFirstClass_before31.pkl'

adID_conType_train_path = 'saved_file/CVR_adID_conType_before28.pkl'
adID_conType_test_path = 'saved_file/CVR_adID_conType_before31.pkl'

positionID_appID_train_path = 'saved_file/CVR_positionID_appID_before28.pkl'
positionID_appID_test_path = 'saved_file/CVR_positionID_appID_before31.pkl'

positionID_advertiseID_train_path = 'saved_file/CVR_positionID_advertiseID_before28.pkl'
positionID_advertiseID_test_path = 'saved_file/CVR_positionID_advertiseID_before31.pkl'
#pos_residence_province
def pos_residence_province_CVR_rate(data):
    positionID = data['positionID'].values[0]
    residence_province = data['residence_province'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([positionID,residence_province,click_amount,conversation_amount,rate])

def get_postionID_connectionType_train():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID','residence_province'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_residence_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','residence_province','pos_residence_click_amount', 'pos_residence_conver_amount','CVR_pos_residence'])
    pickle.dump(data,open(pos_residence_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_postionID_connectionType_test():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))



    train_position_connect = train.groupby(['positionID','residence_province'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_residence_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','residence_province','pos_residence_click_amount', 'pos_residence_conver_amount','CVR_pos_residence'])
    pickle.dump(data,open(pos_residence_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list

#age_hometown_province
def age_hometown_province_CVR_rate(data):
    positionID = data['age_hot'].values[0]
    hometown = data['hometown_province'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([positionID,hometown,click_amount,conversation_amount,rate])

def get_age_hometown_train():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['age_hot','hometown_province'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_hometown_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot','hometown_province','age_hometown_click_amount', 'age_hometown_conver_amount','CVR_age_hometown'])
    pickle.dump(data,open(age_hometown_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_hometown_test():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))



    train_position_connect = train.groupby(['age_hot', 'hometown_province'], as_index=False)
    data_list = train_position_connect.apply(lambda x: age_hometown_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot', 'hometown_province', 'age_hometown_click_amount',
                                                  'age_hometown_conver_amount', 'CVR_age_hometown'])
    pickle.dump(data, open(age_hometown_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


##age_residence
def age_residence_province_CVR_rate(data):
    positionID = data['age_hot'].values[0]
    hometown = data['residence_province'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([positionID,hometown,click_amount,conversation_amount,rate])

def get_age_residence_train():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['age_hot','residence_province'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_residence_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot','residence_province','age_residence_click_amount', 'age_residence_conver_amount','CVR_age_residence'])
    pickle.dump(data,open(age_residence_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_residence_test():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))



    train_position_connect = train.groupby(['age_hot', 'residence_province'], as_index=False)
    data_list = train_position_connect.apply(lambda x: age_residence_province_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot', 'residence_province', 'age_residence_click_amount',
                                                  'age_residence_conver_amount', 'CVR_age_residence'])
    pickle.dump(data, open(age_residence_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


##camgaign_connectionType
def camgaign_connectionType_CVR_rate(data):
    camgaignID = data['camgaignID'].values[0]
    connectionType = data['connectionType'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([camgaignID,connectionType,click_amount,conversation_amount,rate])

def get_camgaign_connectionType_train():
    train = pickle.load(open('saved_file/origin_train_pos_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['camgaignID','connectionType'],as_index=False)
    data_list = train_position_connect.apply(lambda x:camgaign_connectionType_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['camgaignID','connectionType','camgaignID_connect_click_amount', 'camgaignID_connect_conver_amount','CVR_camgaignID_connect'])
    pickle.dump(data,open(camgaign_connectionType_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_camgaign_connectionType_test():
    train = pickle.load(open('saved_file/origin_train_pos_ad.pkl','rb'))



    train_position_connect = train.groupby(['camgaignID', 'connectionType'], as_index=False)
    data_list = train_position_connect.apply(lambda x: camgaign_connectionType_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['camgaignID', 'connectionType', 'camgaignID_connect_click_amount',
                                                  'camgaignID_connect_conver_amount', 'CVR_camgaignID_connect'])
    pickle.dump(data, open(camgaign_connectionType_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


##user_platform
def userID_platform_CVR_rate(data):
    userID = data['userID'].values[0]
    appPlatform = data['appPlatform'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,appPlatform,click_amount,conversation_amount,rate])

def get_userID_platform_train():
    train = pickle.load(open('saved_file/origin_train_pos_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['userID','appPlatform'],as_index=False)
    data_list = train_position_connect.apply(lambda x:userID_platform_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','appPlatform','userID_appPlatform_click_amount', 'userID_appPlatform_conver_amount','CVR_userID_appPlatform'])
    pickle.dump(data,open(userID_appPlatform_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_userID_platform_test():
    train = pickle.load(open('saved_file/origin_train_pos_ad.pkl','rb'))



    train_position_connect = train.groupby(['userID', 'appPlatform'], as_index=False)
    data_list = train_position_connect.apply(lambda x: userID_platform_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID', 'appPlatform', 'userID_appPlatform_click_amount',
                                                  'userID_appPlatform_conver_amount', 'CVR_userID_appPlatform'])
    pickle.dump(data, open(userID_appPlatform_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list
##user_siteset
def userID_siteSet_CVR_rate(data):
    userID = data['userID'].values[0]
    sitesetID = data['sitesetID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,sitesetID,click_amount,conversation_amount,rate])

def get_userID_sitesetID_train():
    train = pickle.load(open('saved_file/origin_train_pos.pkl','rb'))

    train = train[train['click_day'] < split_day]


    train_position_connect = train.groupby(['userID','sitesetID'],as_index=False)
    data_list = train_position_connect.apply(lambda x:userID_siteSet_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','sitesetID','userID_sitesetID_click_amount', 'userID_sitesetID_conver_amount','CVR_userID_sitesetID'])
    pickle.dump(data,open(userID_sitesetID_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_userID_sitesetID_test():
    train = pickle.load(open('saved_file/origin_train_pos.pkl','rb'))


    train_position_connect = train.groupby(['userID', 'sitesetID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: userID_siteSet_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID', 'sitesetID', 'userID_sitesetID_click_amount',
                                                  'userID_sitesetID_conver_amount', 'CVR_userID_sitesetID'])
    pickle.dump(data, open(userID_sitesetID_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list




##age_camgaingn(user-ad)
def age_camgaign_CVR_rate(data):
    userID = data['age_hot'].values[0]
    camgaignID = data['camgaignID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,camgaignID,click_amount,conversation_amount,rate])

def get_age_camgaignID_train():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['age_hot','camgaignID'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_camgaign_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot','camgaignID','age_hot_camgaignID_click_amount', 'age_hot_camgaignID_conver_amount','CVR_age_hot_camgaignID'])
    pickle.dump(data,open(age_hot_camgaignID_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_camgaignID_test():
    train = pickle.load(open('saved_file/origin_train_user_ad.pkl','rb'))



    train_position_connect = train.groupby(['age_hot','camgaignID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: userID_clich_day_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age_hot','camgaignID','age_hot_camgaignID_click_amount', 'age_hot_camgaignID_conver_amount','CVR_age_hot_camgaignID'])
    pickle.dump(data, open(age_hot_camgaignID_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list

##user_click_time(train)
def userID_clich_day_CVR_rate(data):
    userID = data['userID'].values[0]
    click_day = data['click_day'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,click_day,click_amount,conversation_amount,rate])

def get_userID_click_day_train():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['userID','click_day'],as_index=False)
    data_list = train_position_connect.apply(lambda x:userID_clich_day_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','click_day','userID_clickDay_click_amount', 'userID_clickDay_conver_amount','CVR_userID_clickDay'])
    pickle.dump(data,open(userID_clickDay_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_userID_click_day_test():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))


    train_position_connect = train.groupby(['userID', 'click_day'], as_index=False)
    data_list = train_position_connect.apply(lambda x: userID_clich_day_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','click_day','userID_clickDay_click_amount', 'userID_clickDay_conver_amount','CVR_userID_clickDay'])
    pickle.dump(data, open(userID_clickDay_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list

##position_category
def positionID_category_rate(data):
    userID = data['positionID'].values[0]
    category = data['appCategoryFirstClass'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,category,click_amount,conversation_amount,rate])

def get_positionID_category_train():
    train = pickle.load(open('saved_file/origin_train_ad_appCategory.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID','appCategoryFirstClass'],as_index=False)
    data_list = train_position_connect.apply(lambda x:positionID_category_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','appCategoryFirstClass','positionID_appCategoryFirstClass_click_amount', 'positionID_appCategoryFirstClass_conver_amount','CVR_positionID_appCategoryFirstClass'])
    pickle.dump(data,open(positionID_appCategoryFirstClass_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_positionID_category_test():
    train = pickle.load(open('saved_file/origin_train_ad_appCategory.pkl','rb'))



    train_position_connect = train.groupby(['positionID','appCategoryFirstClass'], as_index=False)
    data_list = train_position_connect.apply(lambda x: positionID_category_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','appCategoryFirstClass','positionID_appCategoryFirstClass_click_amount', 'positionID_appCategoryFirstClass_conver_amount','CVR_positionID_appCategoryFirstClass'])
    pickle.dump(data, open(positionID_appCategoryFirstClass_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list

##ad_connectionType
def adID_connectionType_rate(data):
    adID = data['adID'].values[0]
    connectionType = data['connectionType'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([adID,connectionType,click_amount,conversation_amount,rate])

def get_adID_connectionType_train():
    train = pickle.load(open('saved_file/origin_train_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['adID','connectionType'],as_index=False)
    data_list = train_position_connect.apply(lambda x:adID_connectionType_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['adID','connectionType','adID_conType_click_amount', 'adID_conType_conver_amount','CVR_adID_conType'])
    pickle.dump(data,open(adID_conType_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_adID_connectionType_test():
    train = pickle.load(open('saved_file/origin_train_ad.pkl','rb'))



    train_position_connect = train.groupby(['adID','connectionType'], as_index=False)
    data_list = train_position_connect.apply(lambda x: adID_connectionType_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['adID','connectionType','adID_conType_click_amount','adID_conType_conver_amount','CVR_adID_conType'])
    pickle.dump(data, open(adID_conType_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list

##app_position
def positionID_appId_rate(data):
    appID = data['appID'].values[0]
    positionID = data['positionID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([positionID,appID,click_amount,conversation_amount,rate])

def get_positionID_appId_train():
    train = pickle.load(open('saved_file/origin_train_ad.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID','appID'],as_index=False)
    data_list = train_position_connect.apply(lambda x:positionID_appId_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','appID','positionID_appID_click_amount', 'positionID_appID_conver_amount','CVR_positionID_appID'])
    pickle.dump(data,open(positionID_appID_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_positionID_appId_test():
    train = pickle.load(open('saved_file/origin_train_ad.pkl','rb'))



    train_position_connect = train.groupby(['positionID','appID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: positionID_appId_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','appID','positionID_appID_click_amount','positionID_appID_conver_amount','CVR_positionID_appID'])
    pickle.dump(data, open(positionID_appID_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list



##positionID_advertiseID
def positionID_advertiseID_rate(data):
    advertiseID = data['advertiserID'].values[0]
    positionID = data['positionID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label'] == 1])
    rate = conversation_amount / (click_amount + 1e-5)
    return list([positionID, advertiseID, click_amount, conversation_amount, rate])


def get_positionID_advertiseID_train():
    train = pickle.load(open('saved_file/origin_train_ad.pkl', 'rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID', 'advertiserID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: positionID_advertiseID_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID', 'advertiserID', 'positionID_advertiseID_click_amount',
                                                  'positionID_advertiseID_conver_amount', 'CVR_positionID_advertiseID'])
    pickle.dump(data, open(positionID_advertiseID_train_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


def get_positionID_advertiseID_test():
    train = pickle.load(open('saved_file/origin_train_ad.pkl', 'rb'))


    train_position_connect = train.groupby(['positionID', 'advertiserID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: positionID_advertiseID_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID', 'advertiserID', 'positionID_advertiseID_click_amount',
                                                  'positionID_advertiseID_conver_amount', 'CVR_positionID_advertiseID'])
    pickle.dump(data, open(positionID_advertiseID_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


if __name__ == '__main__':
    get_userID_click_day_train()
    get_userID_click_day_test()
    get_age_camgaignID_train()
    get_age_camgaignID_test()
    get_positionID_category_train()
    get_positionID_category_test()
    get_adID_connectionType_train()
    get_adID_connectionType_test()
    get_positionID_appId_train()
    get_positionID_appId_test()
    get_positionID_advertiseID_train()
    get_positionID_advertiseID_test()


