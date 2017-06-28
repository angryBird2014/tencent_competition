import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
train_ffm = '../../output/ffm/train.ffm'
test_ffm = '../../output/ffm/test.ffm'
valid_ffm = '../../output/ffm/valid.ffm'
valid = '../../output/ffm/validation.csv'
field = ['creativeID', 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
         'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'sitesetID', 'positionType',
         'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'hour', 'minute',
         'cate_1', 'cate_2', 'hometown_city', 'hometown_province', 'residence_city', 'residence_province']

ad_features = ['advertiserID', 'camgaignID', 'adID', 'creativeID', 'appID', 'appCategory', 'appPlatform', 'cate_1', ' cate_2']
user_features = ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'hour']
context_features = ['positionID', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator']

table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index




def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices


feature_indices = set()

with open(train_ffm, 'w') as ftr,  open(valid_ffm, 'w') as fva, open(valid, 'w') as fv:
    fv.write('instanceID,label\n')
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        if 27 <= int(row['date']) <= 28:
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        idx = field_index(k)
                        kv = k + ':' + v
                        features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                        feature_indices.add(kv + '\t' + str(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating train.ffm...', e)
            ftr.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

        if int(row['date']) == 29:
            fv.write('{0},{1}\n'.format(str(e), row['label']))
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        idx = field_index(k)
                        kv = k + ':' + v
                        features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                        feature_indices.add(kv + '\t' + str(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating valid.ffm...', e)
            fva.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))


with open(test_ffm, 'w') as fo:
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    # if kv + '\t' + str(getIndices(kv)) in feature_indices:
                    #     features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                    features.append('{0}:{1}:1'.format(idx, getIndices(kv)))

        if t % 100000 == 0:
            print(datetime.now(), 'creating test.ffm...', t)
        fo.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))


fo.close()

