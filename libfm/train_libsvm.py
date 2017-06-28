import pandas as pd
from sklearn.datasets import dump_svmlight_file
import pickle
#df = pd.read_csv("blabla.csv", encoding='utf8')
df= pickle.load(open('../xwd/train_feat.pkl','rb'))
df.fillna(0,inplace =True)
dummy = pd.get_dummies(df)
mat = dummy.as_matrix()
del df
del dummy
y = pickle.load(open('../xwd/train_label.pkl','rb'))
dump_svmlight_file(mat, y, 'train-output.libsvm')
del mat
del y
