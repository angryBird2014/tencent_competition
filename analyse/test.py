import pandas as pd
import pickle
import gc

merge_src_train_dump_path = '../saved_file/merge_src_train_data.pkl'
merge_src_test_dump_path = '../saved_file/merge_src_test_data.pkl'

def time_transfer(value):
    '''

    :param value:int
    :return: dataframe
    '''

    day = 0
    hour = 0
    minute = 0
    second = 0

    timeStr = str(value)
    if len(timeStr) == 8:
        day = int(timeStr[:2])
        hour = int(timeStr[2:4])
        minute = int(timeStr[4:6])
        second = int(timeStr[6:])

    return day,hour,minute,second



def mergeAllData(merge_src_data_dump_path,path):

    data_df = pickle.load(open(merge_src_data_dump_path,'rb'))
    data = data_df[(data_df['click_day'] >= 24)]
    del data_df
    gc.collect()
    user_install = pickle.load(open('../saved_file/user_app_amount.pkl', 'rb'))
    app_install = pickle.load(open('../saved_file/app_user_amount.pkl', 'rb'))

    user_active_amount = pickle.load(open('../saved_file/user_active_amount.pkl','rb'))
    creativeID_active_amount = pickle.load(open('../saved_file/creativeID_active_amount.pkl','rb'))

    user_appID_category = pickle.load(open('../saved_file/user_appID_category.pkl','rb'))

    user_appID_category.drop_duplicates(inplace=True)

    data = pd.merge(data,user_install,how='left',on='userID')
    del user_install
    gc.collect()
    data = pd.merge(data,app_install,how='left',on='appID')
    del app_install
    gc.collect()
    data = pd.merge(data, user_active_amount, how='left', on='userID')
    del user_active_amount
    gc.collect()
    data = pd.merge(data, creativeID_active_amount, how='left', on='creativeID')
    del creativeID_active_amount
    gc.collect()
    data = pd.merge(data,user_appID_category,how='left',on='userID')
    del user_appID_category
    gc.collect()




    split_point = len(data) // 6
    data1 = data[:split_point][:]
    data2 = data[split_point:split_point * 2][:]
    data3 = data[split_point * 2:split_point * 3][:]
    data4 = data[split_point * 3:split_point*4][:]
    data5 = data[split_point*4:split_point*5][:]
    data6 = data[split_point*5:][:]


    pickle.dump(data1, open(path+'1.pkl', 'wb'), protocol=4)
    del data1
    pickle.dump(data2, open(path+'2.pkl', 'wb'), protocol=4)
    del data2
    pickle.dump(data3, open(path+'3.pkl', 'wb'), protocol=4)
    del data3
    pickle.dump(data4, open(path+'4.pkl', 'wb'), protocol=4)
    del data4
    pickle.dump(data5, open(path + '5.pkl', 'wb'), protocol=4)
    del data5
    pickle.dump(data6, open(path + '6.pkl', 'wb'), protocol=4)
    del data6
    #pickle.dump(data_df,open(path,'wb'),protocol=4)



if __name__ == '__main__':


    train_path = '../saved_file/train/'
    test_filename = '../saved_file/test/'
    mergeAllData(merge_src_test_dump_path, test_filename)
    mergeAllData(merge_src_train_dump_path,train_path)
