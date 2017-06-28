import pickle

test_data_path = '../xwd/test_data_2.pkl'

output_test_ffm = '../libffm_part/test.ffm2'
full_dict = '../xwd/full_dict.pkl'


with open(output_test_ffm, 'w') as out:
    train = pickle.load(open(test_data_path, 'rb')).reset_index(drop=True)

    full_dict = pickle.load(open(full_dict, 'rb'))

    column_name = train.columns.tolist()
    for index, row in train.iterrows():
        line = '-1'

        for ix, item in enumerate(column_name):
            k = str(item) + '_' + str(row[item])
            if (ix< 80) and (full_dict.__contains__(k)):
                line += ' ' + str(ix) + ':' + str(full_dict[k]) + ':1'
        out.write(line + '\n')
    out.flush()
