import tensorflow as tf
import numpy as np
import math


class PPGN(object):
    def __init__(self, args, iterator, norm_adj_mat, num_users, num_items_s, num_items_t, is_training):
        self.args = args
        self.iterator = iterator
        self.norm_adj_mat = norm_adj_mat
        self.num_users = num_users
        self.num_items_s = num_items_s
        self.num_items_t = num_items_t
        self.is_training = is_training
        self.n_fold = 100

        self.get_data()
        self.all_weights = self.init_weights()
        self.item_embeddings_s, self.user_embeddings, self.item_embeddings_t = self.creat_gcn_embedd()
        self.inference()
        self.saver = tf.train.Saver(tf.global_variables())


    def get_data(self):
        sample = self.iterator.get_next()
        self.user, self.item_s, self.item_t = sample['user'], sample['item_s'], sample['item_t']
        self.label_s = tf.cast(sample['label_s'], tf.float32)
        self.label_t = tf.cast(sample['label_t'], tf.float32)


    def init_weights(self):
        all_weights = dict()
        initializer = tf.truncated_normal_initializer(0.01)
        regularizer = tf.contrib.layers.l2_regularizer(self.args.regularizer_rate)

        all_weights['user_embeddings'] = tf.get_variable(
            'user_embeddings', (self.num_users, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_s'] = tf.get_variable(
            'item_embeddings_s', (self.num_items_s, self.args.embedding_size), tf.float32, initializer, regularizer)
        all_weights['item_embeddings_t'] = tf.get_variable(
            'item_embeddings_t', (self.num_items_t, self.args.embedding_size), tf.float32, initializer, regularizer)

        self.layers_plus = [self.args.embedding_size] + self.args.gnn_layers

        for k in range(len(self.layers_plus)-1):
            all_weights['W_gc_%d' % k] = tf.get_variable(
                'W_gc_%d'% k, (self.layers_plus[k], self.layers_plus[k+ 1]), tf.float32, initializer, regularizer)
            all_weights['b_gc_%d' % k] = tf.get_variable(
                'b_gc_%d'% k, self.layers_plus[k+ 1], tf.float32, tf.zeros_initializer(), regularizer)
            all_weights['W_bi_%d' % k] = tf.get_variable(
                'W_bi_%d'% k, (self.layers_plus[k], self.layers_plus[k + 1]), tf.float32, initializer, regularizer)
            all_weights['b_bi_%d' % k] = tf.get_variable(
                'b_bi_%d'% k, self.layers_plus[k+ 1], tf.float32, tf.zeros_initializer(), regularizer)

        return all_weights


    def creat_gcn_embedd(self):
        A_fold_hat = self._split_A_hat(self.norm_adj_mat)
        embeddings = tf.concat([self.all_weights['item_embeddings_s'], self.all_weights['user_embeddings'],
                                self.all_weights['item_embeddings_t']], axis=0)
        all_embeddings = [embeddings]

        for k in range(len(self.layers_plus)-1):
            temp_embedd = [tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings) for f in range(self.n_fold)]

            embeddings = tf.concat(temp_embedd, axis=0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.all_weights['W_gc_%d'%k])
                                          + self.all_weights['b_gc_%d'%k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.args.dropout_message)

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, axis=1)
        item_embeddings_s, user_embeddings, item_embeddings_t = tf.split(
            all_embeddings, [self.num_items_s, self.num_users, self.num_items_t], axis=0)

        return item_embeddings_s, user_embeddings, item_embeddings_t


    def _split_A_hat(self, X):
        fold_len = math.ceil((X.shape[0]) / self.n_fold)
        A_fold_hat = [self._convert_sp_mat_to_sp_tensor( X[i_fold*fold_len :(i_fold+1)*fold_len])
                      for i_fold in range(self.n_fold)]

        return A_fold_hat


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.SparseTensor(indices, coo.data, coo.shape)


    def inference(self):
        initializer = tf.truncated_normal_initializer(0.01)
        regularizer = tf.contrib.layers.l2_regularizer(self.args.regularizer_rate)

        with tf.name_scope('embedding'):
            user_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user)
            item_embedding_s = tf.nn.embedding_lookup(self.item_embeddings_s, self.item_s)
            item_embedding_t = tf.nn.embedding_lookup(self.item_embeddings_t, self.item_t)

        with tf.name_scope('propagation'):
            if self.args.NCForMF == 'MF':
                self.logits_dense_s = tf.reduce_sum(tf.multiply(user_embedding, item_embedding_s), 1)
                self.logits_dense_t = tf.reduce_sum(tf.multiply(user_embedding, item_embedding_t), 1)
            elif self.args.NCForMF == 'NCF':
                a_s = tf.concat([user_embedding, item_embedding_s], axis=-1, name='inputs_s')
                a_t = tf.concat([user_embedding, item_embedding_t], axis=-1, name='inputs_t')

                for i, units in enumerate(self.args.mlp_layers):
                    dense_s = tf.layers.dense(a_s, units, tf.nn.relu, kernel_initializer=initializer,
                                          kernel_regularizer = regularizer, name='dense_s_%d' % i)
                    a_s = tf.layers.dropout(dense_s, self.args.dropout_message)

                    dense_t = tf.layers.dense(a_t, units, tf.nn.relu, kernel_initializer=initializer,
                                          kernel_regularizer=regularizer, name='dense_t_%d' % i)
                    a_t = tf.layers.dropout(dense_t, self.args.dropout_message)


                self.logits_dense_s = tf.layers.dense(inputs=a_s,
                                                      units=1,
                                                      kernel_initializer=initializer,
                                                      kernel_regularizer=regularizer,
                                                      name='logits_dense_s')
                self.logits_dense_t = tf.layers.dense(inputs=a_t,
                                                      units=1,
                                                      kernel_initializer=initializer,
                                                      kernel_regularizer=regularizer,
                                                      name='logits_dense_t')
            else:
                raise ValueError

            self.logits_s = tf.squeeze(self.logits_dense_s)
            self.logits_t = tf.squeeze(self.logits_dense_t)

            loss_list_s = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_s, logits=self.logits_s,
                                                                  name='loss_s')
            loss_list_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_t, logits=self.logits_t,
                                                                  name='loss_t')
            loss_w_s = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1.0), lambda: 5.0, lambda: 1.0), self.label_s)
            loss_w_t = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1.0), lambda: 5.0, lambda: 1.0), self.label_t)

            self.loss_s = tf.reduce_mean(tf.multiply(loss_list_s, loss_w_s))
            self.loss_t = tf.reduce_mean(tf.multiply(loss_list_t, loss_w_t))

            self.loss = self.loss_s + self.loss_t

            self.optimizer = tf.train.AdamOptimizer(self.args.lr).minimize(self.loss)

            self.label_replica_s, self.label_replica_t = self.label_s, self.label_t

            _, self.indice_s = tf.nn.top_k(tf.sigmoid(self.logits_s), self.args.topK)
            _, self.indice_t = tf.nn.top_k(tf.sigmoid(self.logits_t), self.args.topK)


    def step(self, sess):
        if self.is_training:
            label_s, indice_s, label_t, indice_t, loss, optim = sess.run(
                [self.label_replica_s, self.indice_s, self.label_replica_t, self.indice_t, self.loss,
                 self.optimizer])

            return loss
        else:
            label_s, indice_s, label_t, indice_t = sess.run(
                [self.label_replica_s, self.indice_s, self.label_replica_t, self.indice_t])
            prediction_s = np.take(label_s, indice_s)
            prediction_t = np.take(label_t, indice_t)

            return prediction_s, label_s, prediction_t, label_t