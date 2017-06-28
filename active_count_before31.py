import pandas as pd
import pickle
origin_train_path = 'saved_file/origin_train.pkl'
user_active_amount_path = 'saved_file/user_active_amount_before31.pkl'
creativeID_active_amount_path = 'saved_file/creativeID_active_amount_before31.pkl'
split_day = 31
day_split = [0,6,12,18,24]

#时间转换为 day,hour,minute
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


def origin_train():
    train = pd.read_csv('data/train.csv')

    clickTime = train['clickTime']
    clickTimeSeries = clickTime.apply(lambda x: time_transfer(x))

    clickTimeDataFrame = pd.DataFrame.from_records(clickTimeSeries.tolist(),
                                                   columns=['click_day', 'click_hour', 'click_minute', 'click_second'])
    data_df = pd.concat([train, clickTimeDataFrame], axis=1)

    pickle.dump(data_df, open('saved_file/origin_train.pkl', 'wb'), protocol=4)

def count(x):
    userID = x['userID']
    first_state_click_amount = len(x[(x['click_hour']>=day_split[0]) & (x['click_hour'] < day_split[1])])
    first_state_conver_amount = len(x[(x['click_hour']>=day_split[0]) & (x['click_hour'] < day_split[1]) & (x['label']==1)])
    first_rate = first_state_conver_amount / (first_state_click_amount+1e-5)
    second_state_click_amount = len(x[(x['click_hour'] >= day_split[1]) & (x['click_hour'] < day_split[2])])
    second_state_conver_amount = len(x[(x['click_hour']>=day_split[1]) & (x['click_hour'] <day_split[2]) & (x['label']==1)])
    seconde_rate = second_state_conver_amount / (second_state_click_amount+1e-5)
    third_state_click_amount = len(x[(x['click_hour'] >= day_split[2]) & (x['click_hour'] < day_split[3])])
    third_state_conver_amount = len(x[(x['click_hour']>=day_split[2]) & (x['click_hour'] <day_split[3]) & (x['label']==1)])
    third_rate = third_state_conver_amount / (third_state_click_amount+1e-5)
    four_state_click_amount = len(x[(x['click_hour'] >= day_split[3]) & (x['click_hour'] < day_split[4])])
    four_state_conver_amount = len(x[(x['click_hour']>=day_split[3]) & (x['click_hour'] <day_split[4]) & (x['label']==1)])
    four_rate = four_state_conver_amount / (four_state_click_amount+1e-5)
    return list([userID.values[0],
                 first_state_click_amount,first_state_conver_amount,first_rate,
                 second_state_click_amount,second_state_conver_amount,seconde_rate,
                 third_state_click_amount,third_state_conver_amount,third_rate,
                 four_state_click_amount,four_state_conver_amount,four_rate])

def user_active_count():

    train = pickle.load(open(origin_train_path,'rb'))

    train_count = train[train['click_day'] < split_day]

    train_count_groupby_user = train_count.groupby('userID',as_index = False)
    train_list = train_count_groupby_user.apply(lambda x: count(x))
    train_list = train_list.values
    train_dataFrame = pd.DataFrame(list(train_list),columns=['userID',
                                                             '0_6_user_click_amount','0_6_user_conver_amount','0_6_user_conver_rate',
                                                             '6_12_user_click_amount','6_12_user_conver_amount','6_12_user_conver_rate',
                                                             '12_18_user_click_amount','12_18_user_conver_amount','12_18_user_conver_rate',
                                                             '18_24_user_click_amount','18_24_user_conver_amount','18_24_user_conver_rate',])
    pickle.dump(train_dataFrame,open(user_active_amount_path,'wb'),protocol=4)


def count_creativeID(x):
    creativeID = x['creativeID']
    first_state_click_amount = len(x[(x['click_hour']>=day_split[0]) & (x['click_hour'] < day_split[1])])
    first_state_conver_amount = len(x[(x['click_hour']>=day_split[0]) & (x['click_hour'] < day_split[1]) & (x['label']==1)])
    first_rate = first_state_conver_amount / (first_state_click_amount+1e-5)
    second_state_click_amount = len(x[(x['click_hour'] >= day_split[1]) & (x['click_hour'] < day_split[2])])
    second_state_conver_amount = len(x[(x['click_hour']>=day_split[1]) & (x['click_hour'] <day_split[2]) & (x['label']==1)])
    seconde_rate = second_state_conver_amount / (second_state_click_amount+1e-5)
    third_state_click_amount = len(x[(x['click_hour'] >= day_split[2]) & (x['click_hour'] < day_split[3])])
    third_state_conver_amount = len(x[(x['click_hour']>=day_split[2]) & (x['click_hour'] <day_split[3]) & (x['label']==1)])
    third_rate = third_state_conver_amount / (third_state_click_amount+1e-5)
    four_state_click_amount = len(x[(x['click_hour'] >= day_split[3]) & (x['click_hour'] < day_split[4])])
    four_state_conver_amount = len(x[(x['click_hour']>=day_split[3]) & (x['click_hour'] <day_split[4]) & (x['label']==1)])
    four_rate = four_state_conver_amount / (four_state_click_amount+1e-5)
    return list([creativeID.values[0],
                 first_state_click_amount,first_state_conver_amount,first_rate,
                 second_state_click_amount,second_state_conver_amount,seconde_rate,
                 third_state_click_amount,third_state_conver_amount,third_rate,
                 four_state_click_amount,four_state_conver_amount,four_rate])

def creativeID_active_count():

    train = pickle.load(open(origin_train_path, 'rb'))

    train_count = train[train['click_day'] < split_day]

    train_count_groupby_user = train_count.groupby('creativeID', as_index=False)
    train_list = train_count_groupby_user.apply(lambda x: count_creativeID(x))
    train_list = train_list.values
    train_dataFrame = pd.DataFrame(list(train_list),columns=['creativeID',
                                                             '0_6_creative_click_amount','0_6_creative_conver_amount','0_6_creative_conver_rate',
                                                             '6_12_creative_click_amount','6_12_creative_conver_amount','6_12_creative_conver_rate',
                                                             '12_18_creative_click_amount','12_18_creative_conver_amount','12_18_creative_conver_rate',
                                                             '18_24_creative_click_amount','18_24_creative_conver_amount','18_24_creative_conver_rate',])
    pickle.dump(train_dataFrame, open(creativeID_active_amount_path, 'wb'),protocol=4)


if __name__ == '__main__':
    #origin_train()
    user_active_count()
    creativeID_active_count()