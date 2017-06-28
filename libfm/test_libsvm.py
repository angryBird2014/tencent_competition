import pandas as pd
from sklearn.datasets import dump_svmlight_file
import pickle
import numpy as np
#df = pd.read_csv("blabla.csv", encoding='utf8')
df= pickle.load(open('../xwd/test_feat.pkl','rb'))
df.fillna(0,inplace =True)
dummy = pd.get_dummies(df)
mat = dummy.as_matrix()

del dummy
y = np.array([0]*len(df))
dump_svmlight_file(mat, y, 'test-output.libsvm')
del df
del mat
del y