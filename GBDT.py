from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pickle
import pandas as pd
#GBDT组合特征
def GBDT_clf(X_train,y_train,X_valid,X_test):
    from sklearn.preprocessing import OneHotEncoder
    #若深度7 ， 树10棵，则共80个叶子节点
    gbdt = GradientBoostingClassifier(learning_rate=0.1,n_estimators=300, max_depth=6)  #10,3
    gbdt_enc = OneHotEncoder()

    gbdt.fit(X_train, y_train)
    gbdt_enc.fit(gbdt.apply(X_train)[:, :, 0])
    # X_train = pd.DataFrame.as_matrix(X_train)   #DataFrame转np.ndarray
    # X_valid = pd.DataFrame.as_matrix(X_valid)
    # X_test = pd.DataFrame.as_matrix(X_test)
    gbdt_train_feature = gbdt_enc.transform(gbdt.apply(X_train)[:, :, 0]).toarray()
    gbdt_valid_feature = gbdt_enc.transform(gbdt.apply(X_valid)[:, :, 0]).toarray()
    gbdt_test_feature = gbdt_enc.transform(gbdt.apply(X_test)[:, :, 0]).toarray()
    return gbdt_train_feature,gbdt_valid_feature,gbdt_test_feature

if __name__ == '__main__':

    train_data = pickle.load(open('xwd/train_feat.pkl','rb'))
    print(len(train_data))
    print(len(train_data.columns.tolist()))
    test_data = pickle.load(open('xwd/test_feat.pkl','rb'))
    train_label = pickle.load(open('xwd/train_label.pkl','rb'))
    valid_data = pickle.load(open('xwd/valid_feat.pkl','rb'))
    valid_label = pickle.load(open('xwd/valid_label.pkl','rb'))
    train_data = train_data.fillna(0)
    valid_data = valid_data.fillna(0)
    test_data = test_data.fillna(0)
    print('load data over...')

    column = ['gbdt_'+str(i) for i in range(19200)]   #80个叶子节点

    gbdt_train_feature,gbdt_valid_feature,gbdt_test_feature = GBDT_clf(
        train_data,train_label,valid_data,test_data)
    gbdt_train_df = pd.DataFrame(gbdt_train_feature, columns=column)
    print(len(gbdt_train_df))
    print(len(gbdt_train_df.columns.tolist()))
    gbdt_valid_df = pd.DataFrame(gbdt_valid_feature, columns=column)
    gbdt_test_df = pd.DataFrame(gbdt_test_feature, columns=column)

    pickle.dump(gbdt_train_df,open('gbdt/gbdt_train.pkl', 'wb'),protocol=4)
    pickle.dump(gbdt_valid_df,open('gbdt/gbdt_valid.pkl', 'wb'),protocol=4)
    pickle.dump(gbdt_test_df,open('gbdt/gbdt_test.pkl', 'wb'),protocol=4)