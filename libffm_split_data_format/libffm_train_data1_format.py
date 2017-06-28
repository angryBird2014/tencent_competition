import pickle

train_data_path = '../xwd/train_data_0.pkl'
train_label = '../xwd/train_label_0.pkl'
output_train_ffm = '../libffm_part/train.ffm0'
full_dict = '../xwd/full_dict.pkl'

# userAmount_0.0
with open(output_train_ffm, 'w') as out:
    train = pickle.load(open(train_data_path, 'rb')).reset_index(drop=True)
    train_label = pickle.load(open(train_label, 'rb')).reset_index(drop=True)
    full_dict = pickle.load(open(full_dict, 'rb'))

    column_name = train.columns.tolist()
    for index, row in train.iterrows():
        line = str(train_label.loc[index])

        for ix, item in enumerate(column_name):
            k = str(item) + '_' + str(row[item])
            if (ix < 80) and (full_dict.__contains__(k)):
                line += ' ' + str(ix) + ':' + str(full_dict[k]) + ':1'
        out.write(line + '\n')
    out.flush()
