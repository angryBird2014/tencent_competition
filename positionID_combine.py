import pandas as pd
import numpy  as np
import pickle
import gc
from functools import partial
import multiprocessing as mp
from joblib import Parallel, delayed

split_day = 28

pos_con_train_path = 'saved_file/CVR_pos_con_before28.pkl'
pos_con_test_path = 'saved_file/CVR_pos_con_before31.pkl'

pos_userID_train_path = 'saved_file/CVR_pos_userID_before28.pkl'
pos_userID_test_path = 'saved_file/CVR_pos_userID_before31.pkl'

userID_conType_train_path = 'saved_file/CVR_userID_conType_before28.pkl'
userID_conType_test_path = 'saved_file/CVR_userID_conType_before31.pkl'

pos_creative_train_path = 'saved_file/CVR_pos_creative_before28.pkl'
pos_creative_test_path = 'saved_file/CVR_pos_creative_before31.pkl'

pos_telOpera_train_path = 'saved_file/CVR_pos_telOperator_before28.pkl'
pos_telOpera_test_path = 'saved_file/CVR_pos_telOperator_before31.pkl'




#pos_con
def pos_con_CVR_rate(data):
    positionID = data['positionID'].values[0]
    connectionType = data['connectionType'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([positionID,connectionType,click_amount,conversation_amount,rate])

def get_postionID_connectionType_train():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))

    train = train[train['click_day'] < split_day]


    train_position_connect = train.groupby(['positionID','connectionType'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_con_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','connectionType','pos_con_click_amount', 'pos_con_conver_amount','CVR_pos_con'])
    pickle.dump(data,open(pos_con_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_postionID_connectionType_test():
    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))


    train_position_connect = train.groupby(['positionID','connectionType'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_con_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['positionID','connectionType','pos_con_click_amount', 'pos_con_conver_amount','CVR_pos_con'])
    pickle.dump(data, open(pos_con_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


#pos_userID
def pos_userID_CVR_rate(data):
    positionID = data['positionID'].values[0]
    userID = data['userID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label'] == 1])
    rate = conversation_amount / (click_amount + 1e-5)
    return list([positionID, userID, click_amount, conversation_amount, rate])

def get_userID_positionID_train():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID','userID'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_userID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'userID', 'pos_user_click_amount', 'pos_user_conver_amount',
                                 'CVR_pos_user'])
    pickle.dump(data, open(pos_userID_train_path, 'wb'), protocol=4)
    del data
    del train
    del data_list


def get_userID_positionID_test():
    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))



    train_position_connect = train.groupby(['positionID','userID'],as_index=False)
    data_list = train_position_connect.apply(lambda x:pos_userID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'userID', 'pos_user_click_amount', 'pos_user_conver_amount',
                                 'CVR_pos_user'])
    pickle.dump(data, open(pos_userID_test_path, 'wb'), protocol=4)
    del data
    del train
    del data_list

#userID_connetType
def userID_connectType_CVR_rate(data):
    connectType = data['connectionType'].values[0]
    userID = data['userID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label'] == 1])
    rate = conversation_amount / (click_amount + 1e-5)
    return list([userID,connectType, click_amount, conversation_amount, rate])

def get_userID_connectType_train():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))
    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['userID', 'connectionType'], as_index=False)

    data_list = train_position_connect.apply(lambda x: userID_connectType_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['userID','connectionType', 'user_con_click_amount', 'user_con_conver_amount',
                                 'CVR_user_con'])

    pickle.dump(data, open(userID_conType_train_path, 'wb'), protocol=4)
    del data_list
    del data
    del train

def get_userID_connectType_test():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))

    train_position_connect = train.groupby(['userID', 'connectionType'], as_index=False)

    data_list = train_position_connect.apply(lambda x: userID_connectType_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['userID', 'connectionType', 'user_con_click_amount', 'user_con_conver_amount',
                                 'CVR_user_con'])

    pickle.dump(data, open(userID_conType_test_path, 'wb'), protocol=4)
    del data_list
    del data
    del train

#pos_creativeID
def pos_creativeID_CVR_rate(data):
    creativeID = data['creativeID'].values[0]
    positionID = data['positionID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label'] == 1])
    rate = conversation_amount / (click_amount + 1e-5)
    return list([positionID, creativeID, click_amount, conversation_amount, rate])

def get_pos_creativeID_train():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))
    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID', 'creativeID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: pos_creativeID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'creativeID', 'pos_creative_click_amount', 'pos_creative_conver_amount',
                                 'CVR_pos_creative'])
    pickle.dump(data, open(pos_creative_train_path, 'wb'), protocol=4)
    del data
    del data_list
    del train

def get_pos_creativeID_test():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))



    train_position_connect = train.groupby(['positionID', 'creativeID'], as_index=False)
    data_list = train_position_connect.apply(lambda x: pos_creativeID_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'creativeID', 'pos_creative_click_amount', 'pos_creative_conver_amount',
                                 'CVR_pos_creative'])
    pickle.dump(data, open(pos_creative_test_path, 'wb'), protocol=4)
    del data
    del data_list
    del train



#pos telecomsOperator
def pos_telOpera_CVR_rate(data):

    tel = data['telecomsOperator'].values[0]
    positionID = data['positionID'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label'] == 1])
    rate = conversation_amount / (click_amount + 1e-5)
    return list([positionID, tel, click_amount, conversation_amount, rate])

def get_pos_telOpera_train():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))
    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['positionID', 'telecomsOperator'], as_index=False)
    data_list = train_position_connect.apply(lambda x: pos_telOpera_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'telecomsOperator', 'pos_tel_click_amount', 'pos_tel_conver_amount',
                                 'CVR_pos_tel'])

    pickle.dump(data, open(pos_telOpera_train_path, 'wb'), protocol=4)
    del data_list
    del data
    del train

def get_pos_telOpera_test():

    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))



    train_position_connect = train.groupby(['positionID', 'telecomsOperator'], as_index=False)
    data_list = train_position_connect.apply(lambda x: pos_telOpera_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list),
                        columns=['positionID', 'telecomsOperator', 'pos_tel_click_amount', 'pos_tel_conver_amount',
                                 'CVR_pos_tel'])

    pickle.dump(data, open(pos_telOpera_test_path, 'wb'), protocol=4)
    del data_list
    del data
    del train



if __name__ == '__main__':


    get_postionID_connectionType_train()
    get_postionID_connectionType_test()
    get_userID_positionID_train()
    get_userID_positionID_test()

    get_userID_connectType_train()
    get_userID_connectType_test()
    get_pos_creativeID_train()
    get_pos_creativeID_test()
    get_pos_telOpera_train()
    get_pos_telOpera_test()

