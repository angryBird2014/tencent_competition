# coding=gbk                                
import numpy as np                          
import pandas as pd                         
import pickle                               
import os                                   
                                            
dirname = 'data/'                           
filename = 'ad.csv'
ad_dump_path = 'saved_file/ad.pkl'

def get_advertiserID_camgaignID():

    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['advertiserID','camgaignID']]
    ad_data_groupby_advertiserID = ad_data.groupby("advertiserID")
    advertiserID_camgaignID = ad_data_groupby_advertiserID.count().reset_index()
    advertiserID_camgaignID = advertiserID_camgaignID.rename(columns = {'camgaignID':'advertiserID_camgaignID_count'})
    advertiserID_camgaignID_mean = advertiserID_camgaignID['advertiserID_camgaignID_count'].mean()
    return advertiserID_camgaignID,int(advertiserID_camgaignID_mean)

def get_advertiserID_appID():

    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['advertiserID','appID']]
    ad_data = ad_data.drop_duplicates()
    ad_data_groupby_advertiserID = ad_data.groupby("advertiserID")
    advertiserID_appID = ad_data_groupby_advertiserID.count().reset_index()
    advertiserID_appID = advertiserID_appID.rename(columns = {'appID':'advertiserID_appID_count'})
    advertiserID_appID_mean = advertiserID_appID['advertiserID_appID_count'].mean()
    return advertiserID_appID,int(advertiserID_appID_mean)

def get_advertiserID_adID():
    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['advertiserID','adID']]
    ad_data_groupby_advertiserID = ad_data.groupby("advertiserID")
    advertiserID_adID = ad_data_groupby_advertiserID.count().reset_index()
    advertiserID_adID = advertiserID_adID.rename(columns={'adID': 'advertiserID_adID_count'})
    advertiserID_adID_mean = advertiserID_adID['advertiserID_adID_count'].mean()
    return advertiserID_adID, int(advertiserID_adID_mean)

def get_advertiserID_creativeID():
    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['advertiserID','creativeID']]
    ad_data_groupby_advertiserID = ad_data.groupby("advertiserID")
    advertiserID_creativeID = ad_data_groupby_advertiserID.count().reset_index()
    advertiserID_creativeID = advertiserID_creativeID.rename(columns={'creativeID': 'advertiserID_creativeID_count'})
    advertiserID_creativeID_mean = advertiserID_creativeID['advertiserID_creativeID_count'].mean()
    return advertiserID_creativeID, int(advertiserID_creativeID_mean)

def get_CamgaignID_adID():
    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['camgaignID','adID']]
    ad_data_groupby_camgaignID = ad_data.groupby("camgaignID")
    camgaignID_adID = ad_data_groupby_camgaignID.count().reset_index()
    camgaignID_adID = camgaignID_adID.rename(columns = {'adID':'camgaignID_adID_count'})
    camgaignID_adID_mean = camgaignID_adID['camgaignID_adID_count'].mean()
    return camgaignID_adID,int(camgaignID_adID_mean)

def get_CamgaignID_creativeID():
    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['camgaignID','creativeID']]
    ad_data_groupby_camgaignID = ad_data.groupby("camgaignID")
    camgaignID_creativeID = ad_data_groupby_camgaignID.count().reset_index()
    camgaignID_creativeID = camgaignID_creativeID.rename(columns = {'creativeID':'camgaignID_creativeID_count'})
    camgaignID_creativeID_mean = camgaignID_creativeID['camgaignID_creativeID_count'].mean()
    return camgaignID_creativeID,int(camgaignID_creativeID_mean)


def get_adID_creativeID():

    ad_data = pd.read_csv(dirname + filename)
    ad_data = ad_data[['adID','creativeID']]
    ad_data_groupby_adID = ad_data.groupby("adID")
    adID_creativeID = ad_data_groupby_adID.count().reset_index()
    adID_creativeID = adID_creativeID.rename(columns = {'creativeID':'adID_creativeID_count'})
    adID_creativeID_mean = adID_creativeID['adID_creativeID_count'].mean()
    return adID_creativeID,int(adID_creativeID_mean)


def AnalyseAPPID():
    ad_data = pd.read_csv(dirname + filename)

    ad_data_groupby_appID = ad_data.groupby("appID")
    ad_groupby_appID = ad_data_groupby_appID.count().reset_index()

    ad = ad_groupby_appID.drop(['camgaignID','adID','advertiserID','appPlatform'],axis = 1)

    ad.rename(columns = {'creativeID':'appID_groupby_count'},inplace = True)

    #这些值作为填充
    ad_groupby_appID_count = ad['appID_groupby_count'].mean()
    '''
    ad_groupby_appID_count_big1 = ad[ad['appID_groupby_count']>1]
    ad_groupby_adID_count = ad_groupby_appID_count_big1['appID_groupby_count'].mean()
    print(ad_groupby_adID_count)
    '''
    return ad,int(ad_groupby_appID_count)

def AnalyseAdID():
    ad_data = pd.read_csv(dirname + filename)

    ad_groupby_adID = ad_data.groupby("adID")

    ad_groupby_adID_count = ad_groupby_adID.count().reset_index()

    ad = ad_groupby_adID_count.drop(['camgaignID', 'advertiserID', 'appID', 'appPlatform'], axis=1)
    ad.rename(columns = {'creativeID':'adID_groupby_count'},inplace = True)

    # 这些值作为填充

    ad_groupby_adID_count = ad['adID_groupby_count'].mean()

    '''
    ad_groupby_adID_count_big_1 = ad[ad['adID_groupby_count'] > 1]
    ad_groupby_adID_count = ad_groupby_adID_count_big_1['adID_groupby_count'].mean()
    print(ad,ad_groupby_adID_count)
    '''
    return ad, int(ad_groupby_adID_count)

def AnalyseCamgaignID():

    ad_data = pd.read_csv(dirname + filename)

    ad_groupby_camgaignID = ad_data.groupby("camgaignID")

    ad_groupby_camgaignID_count = ad_groupby_camgaignID.count().reset_index()

    ad = ad_groupby_camgaignID_count.drop(['adID','advertiserID','appID','appPlatform'],axis = 1)
    ad.rename(columns = {'creativeID':'camgaignID_groupby_count'},inplace = True)

    #这些值作为填充
    creativeID_mean = ad['camgaignID_groupby_count'].mean()

    return ad,int(creativeID_mean)

def getAd():
    ad_data = pd.read_csv(dirname + filename)
    if os.path.exists(ad_dump_path):
        ad_total = pickle.load(open(ad_dump_path,'rb'))
        ad = ad_total[0]
        camgaign_mean = ad_total[1]
        adID_mean = ad_total[2]
        appID_mean = ad_total[3]
        advertiserID_campgignID_mean = ad_total[4]
        advertiserID_adID_mean = ad_total[5]
        advertiserID_creativeID_mean = ad_total[6]
        CamgaignID_adID_mean = ad_total[7]
        CamgaignID_creativeID_mean = ad_total[8]
        adID_creativeID_mean = ad_total[9]
    else:
        #离散
        '''
        appPlatform = ad_data['appPlatform']
        appPlatform_category = pd.get_dummies(appPlatform,prefix="appPlatform")

        ad_data = ad_data.drop('appPlatform',axis=1)
        ad = pd.concat([ad_data,appPlatform_category],axis=1)
        '''
        #不离散
        ad = ad_data

        ad_camgaignID ,camgaign_mean = AnalyseCamgaignID()
        ad_AdID,adID_mean = AnalyseAdID()
        ad_APPID,appID_mean = AnalyseAPPID()

        advertiserID_campagignID, advertiserID_campgignID_mean = get_advertiserID_camgaignID()
        advertiserID_appID, advertiserID_appID_mean = get_advertiserID_appID()
        advertiserID_adID,advertiserID_adID_mean = get_advertiserID_adID()
        advertiserID_creativeID, advertiserID_creativeID_mean = get_advertiserID_creativeID()
        CamgaignID_adID, CamgaignID_adID_mean = get_CamgaignID_adID()
        CamgaignID_creativeID,CamgaignID_creativeID_mean = get_CamgaignID_creativeID()
        adID_creativeID,adID_creativeID_mean = get_adID_creativeID()


        ad = pd.merge(ad,ad_camgaignID,how='left',on='camgaignID')
        ad = pd.merge(ad,ad_AdID,how='left',on='adID')
        ad = pd.merge(ad,ad_APPID,how='left',on='appID')

        ad = pd.merge(ad,advertiserID_campagignID,how='left',on='advertiserID')
        ad = pd.merge(ad,advertiserID_appID,how='left',on='advertiserID')
        ad = pd.merge(ad,advertiserID_adID,how='left',on='advertiserID')
        ad = pd.merge(ad,advertiserID_creativeID,how='left',on='advertiserID')
        ad = pd.merge(ad,CamgaignID_adID,how='left',on='camgaignID')
        ad = pd.merge(ad,CamgaignID_creativeID,how='left',on='camgaignID')
        ad = pd.merge(ad,adID_creativeID,how='left',on='adID')

        # pickle.dump([ad,camgaign_mean,adID_mean,appID_mean,advertiserID_campgignID_mean
        # ,advertiserID_adID_mean,advertiserID_creativeID_mean,CamgaignID_adID_mean,
        #    CamgaignID_creativeID_mean,adID_creativeID_mean], open(ad_dump_path, 'wb'))

        output = open(ad_dump_path,'wb')
        pickle.dump(ad,output)
        output.close()
    # return ad,camgaign_mean,adID_mean,appID_mean,advertiserID_campgignID_mean\
    #     ,advertiserID_adID_mean,advertiserID_creativeID_mean,CamgaignID_adID_mean,\
    #        CamgaignID_creativeID_mean,adID_creativeID_mean

if __name__ == '__main__':

    getAd()
