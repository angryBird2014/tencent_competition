import pandas as pd
import pickle
import gc

if __name__ == '__main__':

    install = pd.read_csv('data/user_installedapps.csv')

    user = pd.read_csv('data/user.csv')

    userID = set(user['userID'].values.tolist())
    data_filterby_user = install.loc[install['userID'].isin(userID)]
    del install
    del user
    gc.collect()
    ad = pd.read_csv('data/ad.csv')

    appID = set(ad['appID'].values.tolist())
    data_filterby_ad = data_filterby_user.loc[data_filterby_user['appID'].isin(appID)]
    del data_filterby_user
    del ad
    gc.collect()
    pickle.dump(data_filterby_ad,open('saved_file/user_install_filter_user_appID.pkl','wb'),protocol=4)
    del data_filterby_ad
