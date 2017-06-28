import pickle
import pandas as pd
output_train_ffm = 'data/train.ffm'
output_valid_ffm = 'data/valid.ffm'
output_test = 'data/test.ffm'
train_data_path = 'data/train_feat.pkl'
train_label_path = 'data/train_label.pkl'
valid_data_path = 'data/valid_feat.pkl'
valid_label_path = 'data/valid_label.pkl'
test_path = 'data/test_feat.pkl'

'''
fileds = ['user','other','adID','appID','advertiserID']


user = ['clickTime','today_repeat_click_at_times','user_cvr_before_today','today_rest_repeat_times',
                'userID','click_hour','click_minute','today_repeat_click_totall_times','userAmount','CVR_age_hometown',
                'age','click_second','residence_province','user_cvr_week_ago_x','CVR_userID_sitesetID',
                'user_click_num_today','CVR_age_appPlatform','user_con_click_amount','user_conv_num_week_ago_x',
                'user_click_num_week_ago_x','residence_city','age_hometown_click_amount','age_residence_click_amount',
                'age_residence_conver_amount','today_click_at_times','education','age_cvr_before_today','hometown_province',
                'CVR_user_con','today_rest_click_times','connectionType','user_click_num_week_ago_y',
                'hometown_city','user_click_num_before_today','gender','age_cvr_week_ago','user_cvr_week_ago_y','age_conv_num_week_ago'
                'age_click_num_week_ago','haveBaby','age_hometown_conver_amount','user_conv_num_before_today','age_click_num_before_today',
                'marriageStatus','telecomsOperator','age_conv_num_before_today','user_conv_num_week_ago_y','user_appInAllApp_num',
                'age_click_num_week_ago','age_conv_num_week_ago']
other = ['repeat_time_interval','CVR_pos_con','CVR_pos_creative','CVR_camgaignID_connect','pos_creative_click_amount',
                 'CVR_appID_age','pos_user_click_amount','CVR_pos_residence','camgaignID_connect_click_amount','pos_creative_conver_amount',
                 'camgaignID_connect_conver_amount','appID_age_conver_amount','CVR_pos_tel','CVR_adID_age','adID_age_click_amount',
                 'CVR_age_residence','appID_age_click_amount','user_firstCater_conn_conv_num','userID_sitesetID_click_amount',
                 'pos_residence_click_amount','pos_residence_conver_amount','user_firstCater_conn_click_num','pos_con_click_amount',
                 'pos_con_conver_amount','pos_tel_conver_amount','user_category_count_0','user_firstCater_conn_cvr','adID_age_conver_amount',
                 'user_category_count_402','user_category_count_104','age_appPlatform_click_amount','user_category_count_203',
                 'user_category_count_405','user_category_count_106','user_category_count_209','age_appPlatform_conver_amount',
                 'user_category_count_401','userID_sitesetID_conver_amount','user_category_count_503',
                 'user_category_count_201','user_category_count_408','user_category_count_105','userAmount_combine_appID_groupby_count',
                 'user_category_count_210','user_category_count_108','user_con_conver_amount','user_category_count_407','user_category_count_301',
                 'user_category_count_103','pos_user_conver_amount','user_category_count_109','user_category_count_303','user_category_count_2',
                 'user_category_count_110','user_DiffAdvertiser_num','user_category_count_403','user_category_count_211','user_category_count_406',
                 'user_category_count_409','CVR_pos_user']
adID = ['positionID_cvr_before_today','positionID','adID']
appID = ['app_install_num_ago_24','appAmount','appID','appID_click_num_before_today','appID_groupby_count','appCategory_count',
         'appID_cvr_before_today','appID_conv_num_before_today','app_user_rate','appCategoryFirstClass','appCategorySecondClass',
         'sitesetID','install_or_not']
advertiserID = ['creative_cvr_week_ago','camgaignID_groupby_count','advertiserID','advertiserID_cvr_before_today',
                 'camgaignID','creativeID','advertiserID_camgaignID_count','advertiser_cvr_week_ago','creative_cvr_before_today',
                 'CVR_creativeID_age','positionID_cvr_week_ago','creative_click_num_week_ago','creative_click_num_before_today',
                 'pos_tel_click_amount','positionType','advertiserID_click_num_before_today','creativeID_age_cick_amount',
                 'advertiser_conv_num_week_ago','positionID_click_num_before_today','advertiserID_conv_num_before_today',
                 'creative_conv_num_week_ago','advertiser_click_num_week_ago','creativeID_age_conver_amount',
                 'positionID_click_num_week_ago','creative_conv_num_before_today','adID_groupby_count','advertiserID_appID_count',
                 'positionID_conv_num_week_ago','positionID_conv_num_before_today']
field_dict = {}
feature_dict = {}


for index,field in enumerate(fileds):
    field_dict[field] = index

def map_feature_field(feature_name):
    if feature_name in user:
        return field_dict['user']
    elif feature_name in other:
        return field_dict['other']
    elif feature_name in adID:
        return field_dict['adID']
    elif feature_name in advertiserID:
        return field_dict['advertiserID']
    elif feature_name in appID:
        return field_dict['appID']


length = 0

for index,feature_in_user in enumerate(user):
    feature_dict[feature_in_user] = index

length = len(user) + length

for index,feature_in_other in enumerate(other):
    feature_dict[feature_in_other] = index + length

length = length + len(other)

for index,feature_in_adID in enumerate(adID):
    feature_dict[feature_in_adID] = index + length

length = length + len(adID)

for index,feature_in_appID in enumerate(appID):
    feature_dict[feature_in_appID] = index + length

length = length + len(appID)

for index,feature_in_advertiserID in enumerate(advertiserID):
    feature_dict[feature_in_advertiserID] = index + length


train_data = pickle.load(open(train_data_path,'rb'))
train_data = train_data.drop(['user_app_rate','user_click_rate_today','user_appInAllApp_rate','user_DiffAdvertiser_rate',
                              'user_category_count_205','user_category_count_204','user_category_count_101',
                              'user_category_count_107','user_category_count_102','user_category_count_1'],axis = 1)
train_data.fillna(0,inplace=True)

columns = train_data.columns.tolist()
train_label = pickle.load(open(train_label_path,'rb'))


valid_data = pickle.load(open(valid_data_path,'rb'))
valid_label = pickle.load(open(valid_label_path,'rb'))

test_data = pickle.load(open(test_path,'rb'))

with open(output_train_ffm,'w') as output:
    for index,row in train_data.iterrows():
        label = train_label.loc[index]
        output.write(str(label))
        print(label)
        for column_name in columns:
            feature_index = feature_dict[column_name]
            field_index = map_feature_field(column_name)
            str_ = '\t' + str(field_index) + ':' + str(feature_index) + ':' + str(row[column_name])
            print(str_)
            output.write(str_)
        output.write('\n')

with open(output_valid_ffm,'w') as output:
    for index,row in valid_data.iterrows():
        label = valid_label.loc[index]
        output.write(str(label))
        print(label)
        for column_name in columns:
            feature_index = feature_dict[column_name]
            field_index = map_feature_field(column_name)
            str_ = '\t' + str(field_index) + ':' + str(feature_index) + ':' + str(row[column_name])
            print(str_)
            output.write(str_)
        output.write('\n')

with open(output_test,'w') as output:
    for index,row in test_data.iterrows():
        label = -1
        output.write(str(label))
        print(label)
        for column_name in columns:
            feature_index = feature_dict[column_name]
            field_index = map_feature_field(column_name)
            str_ = '\t' + str(field_index) + ':' + str(feature_index) + ':' + str(row[column_name])
            print(str_)
            output.write(str_)
        output.write('\n')
'''
train_data = pickle.load(open(train_data_path,'rb'))
valid_data = pickle.load(open(valid_data_path,'rb'))
test_data = pickle.load(open(test_path,'rb'))
fields = train_data.columns.tolist()
full_dict = {}
i= 0

for index,row in train_data.iterrows():
    for item in fields:
        k = str(item) + '_' + str(row[item])
        if full_dict.__contains__(k):
            continue
        else:
            full_dict[k] = i
            i += 1

for index,row in valid_data.iterrows():
    for item in fields:
        k = str(item) + '_' + str(row[item])
        if full_dict.__contains__(k):
            continue
        else:
            full_dict[k] = i
            i += 1
for index,row in test_data.iterrows():
    for item in fields:
        k = str(item) + '_' + str(row[item])
        if full_dict.__contains__(k):
            continue
        else:
            full_dict[k] = i
            i += 1

pickle.dump(full_dict,open('data/full_dict.pkl','wb'),protocol=4)












