import pickle
import compute_loss_and_save
import csv

'''
if __name__ == '__main__':
    with open('./ffm.out') as file:
        rows = file.readlines()
        rows = [float(row) for row in rows]
        compute_loss_and_save.submission(rows)
'''

if __name__ == '__main__':
    train1 = pickle.load(open('data/train_feat.pkl','rb'))
    print(len(train1.columns.tolist()))
    train2 = pickle.load(open('xwd/train_feat.pkl','rb'))
    print(len(train2.columns.tolist()))
