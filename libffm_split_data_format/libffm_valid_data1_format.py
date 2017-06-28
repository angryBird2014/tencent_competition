import pickle

valid_data_path = '../xwd/valid_data_0.pkl'
valid_label = '../xwd/valid_label_0.pkl'
output_valid_ffm = '../libffm_part/valid.ffm0'
full_dict = '../xwd/full_dict.pkl'

# userAmount_0.0
with open(output_valid_ffm, 'w') as out:
    train = pickle.load(open(valid_data_path, 'rb')).reset_index(drop=True)
    train_label = pickle.load(open(valid_label, 'rb')).reset_index(drop=True)
    full_dict = pickle.load(open(full_dict, 'rb'))

    column_name = train.columns.tolist()
    for index, row in train.iterrows():
        line = str(train_label.loc[index])

        for ix, item in enumerate(column_name):
            k = str(item) + '_' + str(row[item])
            if (ix< 80) and (full_dict.__contains__(k)):
                line += ' ' + str(ix) + ':' + str(full_dict[k]) + ':1'
        out.write(line + '\n')
    out.flush()
