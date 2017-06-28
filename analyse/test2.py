import pickle
import pandas as pd
import gc
if __name__ == '__main__':

    '''
    ad = pickle.load(open('../saved_file/ad.pkl','rb'))
    ad = ad.drop(['camgaignID_groupby_count','adID_groupby_count',
                  'appID_groupby_count','advertiserID_camgaignID_count',
                  'advertiserID_appID_count','advertiserID_adID_count',
                  'advertiserID_creativeID_count','camgaignID_adID_count',
                  'camgaignID_creativeID_count','adID_creativeID_count'],axis=1)
    '''
    appCategory = pickle.load(open('../saved_file/app_category.pkl','rb'))
    train_ad = pickle.load(open('../saved_file/origin_train_ad.pkl','rb'))

    train_ad_appCategory = pd.merge(train_ad,appCategory,how='left',on=['appID'])
    print(train_ad_appCategory.columns)
    pickle.dump(train_ad_appCategory,open('../saved_file/origin_train_ad_appCategory.pkl','wb'),protocol=4)
