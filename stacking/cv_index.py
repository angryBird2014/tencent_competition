import pandas as pd

import pickle

from sklearn.model_selection import StratifiedKFold


folds_num = 5
train_label = pickle.load(open('../xwd/train_label.pkl', 'rb'))
train_label = pd.DataFrame.as_matrix(train_label)


cv2 = StratifiedKFold(train_label, n_folds=folds_num)



cv_train_index = []
cv_test_index = []
for k, (train2, test2) in enumerate(cv2):
    cv_train_index.append(train2)
    cv_test_index.append(test2)
    print(type(train2))
    print(train2,test2)
pickle.dump(cv_train_index,open('../xwd/cv_train_index.pkl', 'wb'),protocol=4)
pickle.dump(cv_test_index,open('../xwd/feature_set/cv_test_index.pkl', 'wb'),protocol=4)