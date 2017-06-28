import pandas as pd
from sklearn.datasets import dump_svmlight_file
df = pd.read_csv("blabla.csv", encoding='utf8')
dummy = pd.get_dummies(df)
mat = dummy.as_matrix()
dump_svmlight_file(mat, y, 'svm-output.libsvm')
