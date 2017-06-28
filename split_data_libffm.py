import pickle
import pandas as pd

nums = 3

train_data = pickle.load(open('xwd/train_feat.pkl','rb'))
train_label = pickle.load(open('xwd/train_label.pkl','rb'))
valid_data = pickle.load(open('xwd/valid_feat.pkl','rb'))
valid_label = pickle.load(open('xwd/valid_label.pkl','rb'))
test = pickle.load(open('xwd/test_feat.pkl','rb'))

train_data.fillna(0,inplace=True)

train_split_point = len(train_data)//nums
valid_split_point = len(valid_data) // nums
test_split_point = len(test) // nums
train_pos = 0
valid_pos = 0
test_pos = 0
for index in range(nums-1) :

    train_data_part = train_data[train_pos:train_pos + train_split_point]
    print(train_data_part.index.values)
    pickle.dump(train_data_part,open('xwd/train_data_{0}.pkl'.format(index),'wb'),protocol=4)
    del train_data_part

    train_label_part = train_label[train_pos:train_pos + train_split_point]

    pickle.dump(train_label_part,open('xwd/train_label_{0}.pkl'.format(index), 'wb'), protocol=4)
    del train_label_part

    valid_data_part = valid_data[valid_pos:valid_pos + valid_split_point]
    print(valid_data_part.index.values)
    pickle.dump(valid_data_part,open('xwd/valid_data_{0}.pkl'.format(index), 'wb'), protocol=4)
    del valid_data_part

    valid_label_part = valid_label[valid_pos:valid_pos + valid_split_point]

    pickle.dump(valid_label_part,open('xwd/valid_label_{0}.pkl'.format(index), 'wb'), protocol=4)
    del valid_label_part

    test_part = test[test_pos :test_pos +  test_split_point]
    print(test_part.index.values)
    pickle.dump(test_part,open('xwd/test_part_{0}.pkl'.format(index),'wb'),protocol=4)
    del test_part

    train_pos = train_pos + train_split_point
    test_pos = test_pos + test_split_point
    valid_pos = valid_pos + valid_split_point

train_data_part = train_data[train_pos:]
print(train_data_part.index.values)
pickle.dump(train_data_part,open('xwd/train_data_{0}.pkl'.format(nums-1),'wb'),protocol=4)
del train_data_part

train_label_part = train_label[train_pos:]

pickle.dump(train_label_part,open('xwd/train_label_{0}.pkl'.format(nums-1),'wb'),protocol=4)
del train_label_part

valid_data_part = valid_data[valid_pos:]
print(valid_data_part.index.values)
pickle.dump(valid_data_part,open('xwd/valid_data_{0}.pkl'.format(nums-1),'wb'),protocol=4)
del valid_data_part

valid_label_part = valid_label[valid_pos:]

pickle.dump(valid_label_part,open('xwd/valid_label_{0}.pkl'.format(nums-1),'wb'),protocol=4)
del valid_label_part


test_part = test[test_pos:]
print(test_part.index.values)
pickle.dump(test_part,open('xwd/test_part_{0}.pkl'.format(nums-1),'wb'),protocol=4)
del test_part


