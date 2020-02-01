import os
import numpy as np
import pandas as pd
from itertools import count
from collections import defaultdict
from multiprocessing import Pool


class Dataset:
    def __init__(self, file_path, args):
        self.args = args
        self.path = '/'.join(file_path.split('/')[:-1])
        self.name = file_path.split('/')[-1].split('_')[0]
        self.train_npy_path = file_path.replace('.csv', '_train.npy')
        self.test_npy_path = file_path.replace('.csv', '_test.npy')

        if not(os.path.exists(self.train_npy_path) and os.path.exists(self.test_npy_path)):
            self._bulid_data(file_path)
        else:
            if args.data_rebuild == True:
                self._bulid_data(file_path)
            else:
                df, self.num_users, self.num_items = self._load_data(file_path)
                # self.train_df = pd.read_csv(self.path+'/%s_train_df.csv'% self.name)
                # self.test_df = pd.read_csv(self.path+'/%s_test_df.csv'% self.name)
                self.pos_dict = self._construct_pos_dict(df)
                self.train_dict = np.load(self.train_npy_path).item()
                self.test_dict = np.load(self.test_npy_path).item()


    def _bulid_data(self, file_path):
        df, self.num_users, self.num_items = self._load_data(file_path)
        self.pos_dict = self._construct_pos_dict(df)
        self.train_df, self.test_df = self._split_train_test(df)
        self.train_dict = self._construct_train(self.train_df)
        self.test_dict = self._construct_test(self.test_df)


    def _load_data(self, file_path):
        df = pd.read_csv(file_path, sep=',', usecols=[0, 1])

        # constructing index
        uiterator = count(0)
        udict = defaultdict(lambda: next(uiterator))
        [udict[user] for user in sorted(df['reviewerID'].tolist())]
        iiterator = count(0)
        idict = defaultdict(lambda: next(iiterator))
        [idict[item] for item in sorted(df['asin'].tolist())]

        self.udict = udict
        self.idict = idict

        df['uidx'] = df['reviewerID'].map(lambda x: udict[x])
        df['iidx'] = df['asin'].map(lambda x: idict[x])
        del df['reviewerID'], df['asin']
        print('Load %s data successfully with %d users, %d products and %d interactions.'
              %(self.name, len(udict), len(idict), df.shape[0]))

        return df, len(udict), len(idict)


    def _construct_pos_dict(self, df):
        # we can't build a negative dictionary cause it'll cost huge memory
        pos_dict = defaultdict(set)
        for user, item in zip(df['uidx'], df['iidx']):
            pos_dict[user].add(item)

        return pos_dict


    def _split_train_test(self, df):
        test_list = []
        print('Spliting data of train and test...')
        with Pool(self.args.processor_num) as pool:
            nargs = [(user, df, self.args.test_size) for user in range(self.num_users)]
            test_list = pool.map(self._split, nargs)

        test_df = pd.concat(test_list)
        train_df = df.drop(test_df.index)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(self.path+'/%s_train_df.csv'% self.name, index=False)
        test_df.to_csv(self.path+'/%s_test_df.csv'% self.name, index=False)

        return train_df, test_df


    def _construct_train(self, df):
        # It's desperate to use df to calculate... so slow!!!
        print('Adding negative data to train_df...')
        users = []
        items = []
        labels = []
        with Pool(self.args.processor_num) as pool:
            nargs = [(user, item, self.num_items, self.pos_dict, self.args.train_neg_num, True)
                     for user, item in zip(df['uidx'], df['iidx'])]
            res_list = pool.map(self._add_negtive, nargs)

        for (batch_users, batch_items, batch_labels) in res_list:
            users += batch_users
            items += batch_items
            labels += batch_labels

        data_dict = {'user': users, 'item': items, 'label': labels}
        np.save(self.train_npy_path, data_dict)

        return data_dict


    def _construct_test(self, df):
        print('Adding negative data to test_df...')
        users = []
        items = []
        labels = []

        with Pool(self.args.processor_num) as pool:
            nargs = [(user, item, self.num_items, self.pos_dict, self.args.test_neg_num, False)
                     for user, item in zip(df['uidx'], df['iidx'])]
            res_list = pool.map(self._add_negtive, nargs)

        for batch_users, batch_items, batch_labels in res_list:
            users += batch_users
            items += batch_items
            labels += batch_labels

        data_dict = {'user': users, 'item': items, 'label': labels}
        np.save(self.test_npy_path, data_dict)

        return data_dict


    # The 2 functions below are designed for multiprocessing task
    @staticmethod
    def _split(args):
        user, df, test_size = args
        sample_test = df[df['uidx'] == user].sample(n=test_size)

        return sample_test


    @staticmethod
    def _add_negtive(args):
        user, item, num_items, pos_dict, neg_num, train = args
        users, items, labels = [], [], []
        neg_set = set(range(num_items)).difference(pos_dict[user])
        neg_sample_list = np.random.choice(list(neg_set), neg_num, replace=False).tolist()
        for neg_sample in neg_sample_list:
            users.append(user)
            items.append(neg_sample)
            labels.append(0) if train == True else labels.append(neg_sample)

        users.append(user)
        items.append(item)
        if train == True:
            labels.append(1)
        else:
            labels.append(item)

        return (users, items, labels)