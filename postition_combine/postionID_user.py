import pandas as pd
import numpy  as np
import pickle

split_day = 28

train_path = 'saved_file/CVR_pos_userID_before28.pkl'
test_path = 'saved_file/CVR_pos_userID_before31.pkl'

#
def get_click_rate(data):

    data['CVR_position_userID'] = len(data[data['label']==1])/len(data)
    return data

def get_userID_positionID_train():
    train = pickle.load(open('saved_file/origin_train.pkl','rb'))

    train = train[train['click_day'] < split_day]
    train_position_connect = train.groupby(['positionID','connectionType'],as_index=False)
    data = train_position_connect.apply(lambda x:get_click_rate(x))
    print(data.head())
    pickle.dump(data,open(train_path,'wb'),protocol=4)

def get_userID_positionID_test():
    train = pickle.load(open('saved_file/origin_train.pkl', 'rb'))
    train_position_connect = train.groupby(['positionID', 'connectionType'], as_index=False)
    data = train_position_connect.apply(lambda x: get_click_rate(x))
    print(data.head())
    pickle.dump(data, open(test_path, 'wb'), protocol=4)


if __name__ == '__main__':
    get_userID_positionID_train()
    get_userID_positionID_test()
