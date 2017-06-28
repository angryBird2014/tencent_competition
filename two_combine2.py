import pandas as pd
import numpy  as np
import pickle



split_day = 28



age_clickTime_train_path = 'saved_file/CVR_age_clickTime_before28.pkl'
age_clickTime_test_path = 'saved_file/CVR_age_clickTime_before31.pkl'

userID_clickTime_train_path = 'saved_file/CVR_userID_clickTime_before28.pkl'
userID_clickTime_test_path = 'saved_file/CVR_userID_clickTime_before31.pkl'



#age_clickTime
def age_clickTime_CVR_rate(data):
    age = data['age'].values[0]
    click_hour = data['clickTime'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([age,click_hour,click_amount,conversation_amount,rate])

def get_age_clickTime_train():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['age','clickTime'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_clickTime_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age','clickTime','age_clickTime_click_amount', 'age_clickTime_conver_amount','CVR_age_clickTime'])
    pickle.dump(data,open(age_clickTime_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_age_clickTime_test():
    train = pickle.load(open('saved_file/origin_train_user.pkl','rb'))



    train_position_connect = train.groupby(['age','clickTime'],as_index=False)
    data_list = train_position_connect.apply(lambda x:age_clickTime_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['age','clickTime','age_clickTime_click_amount', 'age_clickTime_conver_amount','CVR_age_clickTime'])
    pickle.dump(data,open(age_clickTime_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list




#userID_clickTime
def userID_clickTime_CVR_rate(data):
    userID = data['userID'].values[0]
    clickTime = data['clickTime'].values[0]
    click_amount = len(data)
    conversation_amount = len(data[data['label']==1])
    rate = conversation_amount / (click_amount+1e-5)
    return list([userID,clickTime,click_amount,conversation_amount,rate])

def get_userID_clickTime_train():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))

    train = train[train['click_day'] < split_day]



    train_position_connect = train.groupby(['userID','clickTime'],as_index=False)
    data_list = train_position_connect.apply(lambda x:userID_clickTime_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','clickTime','userID_clickTime_click_amount', 'userID_clickTime_conver_amount','CVR_userID_clickTime'])
    pickle.dump(data,open(userID_clickTime_train_path,'wb'),protocol=4)
    del data
    del train
    del data_list

def get_userID_clickTime_test():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))



    train_position_connect = train.groupby(['userID','clickTime'],as_index=False)
    data_list = train_position_connect.apply(lambda x:userID_clickTime_CVR_rate(x))
    data_list = data_list.values
    data = pd.DataFrame(list(data_list), columns=['userID','clickTime','userID_clickTime_click_amount', 'userID_clickTime_conver_amount','CVR_userID_clickTime'])
    pickle.dump(data,open(userID_clickTime_test_path,'wb'),protocol=4)
    del data
    del train
    del data_list




if __name__ == '__main__':
    get_age_clickTime_train()
    get_age_clickTime_test()
    get_userID_clickTime_train()
    get_userID_clickTime_test()




