import numpy as np
import scipy.sparse as sp
import os, sys, time
from tqdm import tqdm
sys.path.append("..")
from model import PPGN
from multiprocessing import Pool
import tensorflow as tf
from utils import metrics


def train(dataset_s, dataset_t, args):
    with tf.Session() as sess:
        train_path = dataset_s.path + '/cross_' + '_'.join([dataset_s.name, dataset_t.name]) + '_train.npy'
        test_path = dataset_t.path + '/cross_' + '_'.join([dataset_s.name, dataset_t.name]) + '_test.npy'
        if os.path.exists(train_path) and os.path.exists(test_path) and args.cross_data_rebuild == False:
            print('Loading cross data..')
            train_dict = np.load(train_path).item()
            test_dict = np.load(test_path).item()
        else:
            print('Building cross data..')
            train_dict, test_dict = cross_data_build(dataset_s, dataset_t, args, train_path, test_path)
        print('Get cross data successfully.')

        norm_adj_mat = load_mat(dataset_s, dataset_t, args)

        print('Loading train data from train_dict...')
        train_data = tf.data.Dataset.from_tensor_slices(train_dict)
        train_data = train_data.shuffle(buffer_size=len(train_dict['user'])).batch(args.batch_size)
        print('Loading test data from test_dict...')
        test_data = tf.data.Dataset.from_tensor_slices(test_dict)
        # Test data doesn't need to be shuffled, the first item of every (test_size+test_neg_num) is the positive item.
        test_data = test_data.batch(args.test_size + args.test_neg_num)

        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

        model = PPGN(args, iterator, norm_adj_mat, dataset_s.num_users, dataset_s.num_items, dataset_t.num_items, True)

        print("Creating model with fresh parameters...")
        sess.run(tf.global_variables_initializer())

        count = 0
        loss = 0
        last_count = 0
        hr_s_list, mrr_s_list, ndcg_s_list = [], [], []
        hr_t_list, mrr_t_list, ndcg_t_list = [], [], []
        for epoch in range(1, args.epochs + 1):
            print('=' * 30 + ' EPOCH %d ' % epoch + '=' * 30)
            ################################## TRAINING ################################
            if 6 > epoch > 3:
                model.args.lr = 1e-4
            if epoch >= 6:
                model.args.lr = 1e-5
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            start_time = time.time()

            try:
                while True:
                    count += 1
                    loss += model.step(sess)
                    if count % 1000 == 0:
                        print('Epoch %d, step %d, with average loss of %.4f in last %d steps;'
                              % (epoch, count, loss / (count - last_count), count - last_count))
                        loss = 0
                        last_count = count
            except tf.errors.OutOfRangeError:
                print("Epoch %d, finish training " % epoch + "took " +
                      time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)) + ';')

            ################################## TESTING ################################
            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            start_time = time.time()
            HR_s, MRR_s, NDCG_s = [], [], []
            HR_t, MRR_t, NDCG_t = [], [], []
            predictions_s, labels_s, predictions_t, labels_t = model.step(sess)

            cnt = 1
            try:
                while True:
                    predictions_s, labels_s, predictions_t, labels_t = model.step(sess)
                    hr_s, mrr_s, ndcg_s = evaluate(predictions_s, labels_s)
                    hr_t, mrr_t, ndcg_t = evaluate(predictions_t, labels_t)
                    HR_s.append(hr_s)
                    MRR_s.append(mrr_s)
                    NDCG_s.append(ndcg_s)
                    HR_t.append(hr_t)
                    MRR_t.append(mrr_t)
                    NDCG_t.append(ndcg_t)
                    cnt += 1
            except tf.errors.OutOfRangeError:
                hr_s = np.array(HR_s).mean()
                mrr_s = np.array(MRR_s).mean()
                ndcg_s = np.array(NDCG_s).mean()
                hr_t = np.array(HR_t).mean()
                mrr_t = np.array(MRR_t).mean()
                ndcg_t = np.array(NDCG_t).mean()
                hr_s_list.append(hr_s)
                mrr_s_list.append(mrr_s)
                ndcg_s_list.append(ndcg_s)
                hr_t_list.append(hr_t)
                mrr_t_list.append(mrr_t)
                ndcg_t_list.append(ndcg_t)
                print("Epoch %d, finish testing " % epoch + "took: " +
                      time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)) + ';')
                print('Epoch %d, %s HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
                      (epoch, dataset_s.name, hr_s, mrr_s, ndcg_s))
                print('Epoch %d, %s HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
                      (epoch, dataset_t.name, hr_t, mrr_t, ndcg_t))

        print('=' * 30 + 'Finish training' + '=' * 30)
        print('%s best HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
              (dataset_s.name, max(hr_s_list), max(mrr_s_list), max(ndcg_s_list)))
        print('%s best HR is %.4f, MRR is %.4f, NDCG is %.4f;' %
              (dataset_t.name, max(hr_t_list), max(mrr_t_list), max(ndcg_t_list)))


def evaluate(predictions, labels):
    label = int(labels[-1])
    hr = metrics.hit(label, predictions)
    mrr = metrics.mrr(label, predictions)
    ndcg = metrics.ndcg(label, predictions)

    return hr, mrr, ndcg


def cross_data_build(dataset_s, dataset_t, args, train_path, test_path):
    # multiprocessing
    with Pool(args.processor_num) as pool:
        nargs = [(user, dataset_s.pos_dict, dataset_t.pos_dict, dataset_s.num_items, dataset_t.num_items,
                  args.train_neg_num) for user in range(dataset_s.num_users)]
        extend_list = pool.map(_cross_build, nargs)

    for (extend_users, extend_items, extend_labels, flag) in extend_list:
        if flag == 't':
            dataset_t.train_dict['user'].extend(extend_users)
            dataset_t.train_dict['item'].extend(extend_items)
            dataset_t.train_dict['label'].extend(extend_labels)
        elif flag == 's':
            dataset_s.train_dict['user'].extend(extend_users)
            dataset_s.train_dict['item'].extend(extend_items)
            dataset_s.train_dict['label'].extend(extend_labels)

    train_dict_s, test_dict_s = dataset_s.train_dict, dataset_s.test_dict
    train_dict_t, test_dict_t = dataset_t.train_dict, dataset_t.test_dict

    q_s = np.argsort(np.array(train_dict_s['user']))
    q_t = np.argsort(np.array(train_dict_t['user']))

    users_s = np.array(train_dict_s['user'])[q_s].tolist()
    users_t = np.array(train_dict_t['user'])[q_t].tolist()
    assert users_s == users_t
    users = users_s

    items_s = np.array(train_dict_s['item'])[q_s].tolist()
    labels_s = np.array(train_dict_s['label'])[q_s].tolist()

    items_t = np.array(train_dict_t['item'])[q_t].tolist()
    labels_t = np.array(train_dict_t['label'])[q_t].tolist()

    train_dict = {'user': users, 'item_s': items_s, 'item_t': items_t,'label_s': labels_s, 'label_t':labels_t}
    np.save(train_path, train_dict)

    assert test_dict_s['user'] == test_dict_t['user']
    test_dict = {'user': test_dict_s['user'], 'item_s': test_dict_s['item'], 'item_t': test_dict_t['item'],
                 'label_s': test_dict_s['label'], 'label_t':test_dict_t['label']}
    np.save(test_path, test_dict)

    return train_dict, test_dict


def _cross_build(args):
    user, posdict_s, posdict_t ,num_items_s, num_items_t, per_neg_num = args

    num_item_s = len(posdict_s[user])
    num_item_t = len(posdict_t[user])
    if num_item_s > num_item_t:
        flag = 't'
        pos_num = num_item_s - num_item_t
        neg_num = per_neg_num * pos_num
        pos_set = set(posdict_t[user])
        neg_set = set(range(num_items_t)) - pos_set
    elif num_item_t > num_item_s:
        flag = 's'
        pos_num = num_item_t-num_item_s
        neg_num = per_neg_num * pos_num
        pos_set = set(posdict_s[user])
        neg_set = set(range(num_items_s)) - pos_set
    else:
        return [],[],[],''

    extend_users = (pos_num + neg_num)*[user]
    extend_items_neg = np.random.choice(list(neg_set), neg_num, replace=True)
    extend_items_pos = np.random.choice(list(pos_set), pos_num, replace=True)
    extend_items = np.concatenate([extend_items_neg, extend_items_pos])
    extend_labels = neg_num*[0] + pos_num*[1]

    return extend_users, extend_items, extend_labels, flag


def load_mat(dataset_s, dataset_t, args):
    norm_adj_path = '%s/cross_%s_%s_norm_adj_mat.npz'% (dataset_s.path, dataset_s.name, dataset_t.name)
    if os.path.exists(norm_adj_path) and args.mat_rebuild == False:
        print('Loading adjacent mats...')
        norm_adj_mat = sp.load_npz(norm_adj_path)
    else:
        print('Building adjacent mats..')
        num_users = dataset_s.num_users
        num_items_s = dataset_s.num_items
        num_items_t = dataset_t.num_items

        train_df_s = {'user':dataset_s.train_dict['user'][args.train_neg_num::args.train_neg_num+1],
                      'item':dataset_s.train_dict['item'][args.train_neg_num::args.train_neg_num+1]}
        train_df_t = {'user':dataset_t.train_dict['user'][args.train_neg_num::args.train_neg_num+1],
                      'item':dataset_t.train_dict['item'][args.train_neg_num::args.train_neg_num+1]}

        R_s = sp.dok_matrix((num_users, num_items_s), dtype=np.float32)
        R_t = sp.dok_matrix((num_users, num_items_t), dtype=np.float32)

        for user, item in zip(train_df_s['user'], train_df_s['item']):
            R_s[user, item] = 1.0

        for user, item in zip(train_df_t['user'], train_df_t['item']):
            R_t[user, item] = 1.0

        R_s, R_t = R_s.tolil(), R_t.tolil()

        plain_adj_mat = sp.dok_matrix((num_items_s+ num_users+ num_items_t, num_items_s+ num_users+ num_items_t),
                                      dtype=np.float32).tolil()
        plain_adj_mat[num_items_s: num_items_s+ num_users, :num_items_s] = R_s
        plain_adj_mat[:num_items_s, num_items_s: num_items_s+ num_users] = R_s.T
        plain_adj_mat[num_items_s: num_items_s+ num_users, num_items_s+ num_users:] = R_t
        plain_adj_mat[num_items_s+ num_users:, num_items_s: num_items_s+ num_users] = R_t.T
        plain_adj_mat = plain_adj_mat.todok()

        norm_adj_mat = normalized_adj_single(plain_adj_mat+ sp.eye(plain_adj_mat.shape[0]))

        sp.save_npz(norm_adj_path, norm_adj_mat)

    print('Get adjacent mats successfully.')

    return norm_adj_mat


def normalized_adj_single(adj):
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj