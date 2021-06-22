# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 15:11
# @Author : lee
# @FileName : train.py

import argparse
import os, time
import tensorflow as tf
from LoadData import parse_function_
from Logging import Logging
from Pack import Model
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_fscore_support,precision_recall_curve
import numpy as np
a = 1
def eval_epoch(args, sess, test_score, test_loss, test_data,a1_,a2_,a3_,test=False):
    global a
    loss = []
    score = []
    pred = []
    label = []
    a1,a2,a3,ta = [],[],[],[]
    user = []
    while True:
        try:
            score_, loss_, label_,label2_,a1__,a2__,a3__,u= sess.run([test_score, test_loss, test_data['label'],test_data['label2'],a1_,a2_,a3_,test_data['user']])
            loss.append(loss_)
            score.extend(score_)
            label.extend(label_)
            a1.extend(a1__)
            a2.extend(a2__)
            a3.extend(a3__)
            user.extend(u)
        except tf.errors.OutOfRangeError:
            break

    auc = roc_auc_score(label, score)
    pred = list(map(lambda x: 1 if x>=0.5 else 0, score))
    acc = accuracy_score(label, pred)
    
    prec, rec, f1, _ = precision_recall_fscore_support(label, pred, average="binary")
    if test:
        file = open('../result/'+args.dataset+'.txt','a')
        file.write(str(args)+'\n')
        file.write(str(auc)+'\n')
        # np.save('../result/'+args.dataset+'_user'+str(a)+'_'+str(args.dimension)+'.npy',user)
        # np.save('../result/'+args.dataset+'_att'+str(a)+'_'+str(args.dimension)+'.npy',a1)
        # a+=1
        # file.write(str(a1)+'\n\n')
    return np.mean(np.array(loss)), auc, f1, acc, prec, rec

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../Data/', help='Input data path.')
    parser.add_argument('--dataset', type=str, default='3day', help='Dataset.')
    parser.add_argument('--epoch', type=int, default=20, help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--K', type=int, default=4, help='Disentangle components.')
    parser.add_argument('--reg', type=float, default=1e-5, help='Regularization.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--drop_out', type=float, default=0.4, help='dropout rate.')
    parser.add_argument('--dimension', type=int, default=64, help='Dimension of embedding size.')
    parser.add_argument('--gpu_id', type=int, default=0, help='Gpu id.')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Buffer size.')
    parser.add_argument('--ratio', type=float, default=1, help='train ratio.')
    args = parser.parse_args()

    
    # train_f1
    if args.dataset == '3day':
        print('3day')
        f_max_len = 20
        u_max_pack = 50
        pack_max_nei_b = 20
        pack_max_nei_f = 20
        n_users = 554237  
        n_items = 344087  
        n_bizs =166465
        u_max_i = 99 # 71
        u_max_f = 220
        n_train = 1990000
        test_filenames = '../data/test_f1.tfrecords'
        train_filenames = '../data/train_f1.tfrecords'
        valid_filenames = '../data/validation_f1.tfrecords'
        
    
    elif args.dataset == '5day':
        print('5day')
        f_max_len = 20
        u_max_pack = 50
        pack_max_nei_b = 20
        pack_max_nei_f = 20
        n_users = 778996  
        n_items = 555328  
        n_bizs = 222618
        u_max_i = 174
        u_max_f = 253
        n_train = 3710000
        test_filenames = '../data/test_f2.tfrecords'
        train_filenames = '../data/train_f2.tfrecords'
        valid_filenames = '../data/validation_f2.tfrecords'
    
    else:
        print('10day')
        f_max_len = 20
        u_max_pack = 50
        pack_max_nei_b = 20
        pack_max_nei_f = 20
        n_users = 1265569   
        n_items = 1321274   
        n_bizs = 390542
        u_max_i = 273
        u_max_f = 289
        n_train = 7910000
        test_filenames = '../data/test_f3.tfrecords'
        train_filenames = '../data/train_f3.tfrecords'
        valid_filenames = '../data/validation_f3.tfrecords'
    


    padded_shape = {'user': [],
                'item': [],
                'biz': [],
                'friends': [f_max_len],
                'user_items': [u_max_i],
                'user_bizs': [u_max_i],
                'user_friends': [u_max_f],
                'user_packages': [u_max_pack, 2+f_max_len],
                'pack_neighbors_b': [pack_max_nei_b, 2+f_max_len],
                'pack_neighbors_f': [pack_max_nei_f, 2+f_max_len],
                'label': [],'label2': []}

    # --------------------------read data from files---------------------------------
    train_dataset = tf.data.TFRecordDataset(train_filenames)
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    valid_dataset = tf.data.TFRecordDataset(valid_filenames)
    # -----------shuffle and padding of datasets-----------------------
    train_dataset = train_dataset.map(parse_function_(f_max_len)).shuffle(buffer_size=args.buffer_size)
    test_dataset = test_dataset.map(parse_function_(f_max_len))
    valid_dataset = valid_dataset.map(parse_function_(f_max_len))

    train_batch_padding_dataset = train_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,
                                                             drop_remainder=True)
    train_iterator = train_batch_padding_dataset.make_initializable_iterator()
    test_batch_padding_dataset = test_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,
                                                           drop_remainder=True)
    test_iterator = test_batch_padding_dataset.make_initializable_iterator()
    valid_batch_padding_dataset = valid_dataset.padded_batch(args.batch_size, padded_shapes=padded_shape,
                                                             drop_remainder=True)
    valid_iterator = valid_batch_padding_dataset.make_initializable_iterator()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    time0 = time.time()
    log_path = '../Log/%s_%s.log' % (args.dataset, str(int(time0)))
    log = Logging(log_path)

    log.print('Initializing model...')
    model = Model(args, n_users, n_items, n_bizs, f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f, u_max_i, u_max_f)

    train_data = train_iterator.get_next()
    test_data = test_iterator.get_next()
    valid_data = valid_iterator.get_next()

    with tf.variable_scope('model', reuse=None):
        train_loss, train_score, train_opt, att, u_e, u, p= model.forward(train_data['user'], train_data['item'], train_data['biz'],
                                              train_data['friends'], train_data['user_items'],
                                              train_data['user_bizs'],train_data['user_friends'],
                                              train_data['user_packages'], train_data['pack_neighbors_b'],
                                              train_data['pack_neighbors_f'], train_data['label'], train_data['label2'], train=True)

    with tf.variable_scope('model', reuse=True):
        test_loss, test_score, a1,a2,a3 = model.forward(test_data['user'], test_data['item'], test_data['biz'],
                                              test_data['friends'], test_data['user_items'],
                                              test_data['user_bizs'],test_data['user_friends'],
                                              test_data['user_packages'], test_data['pack_neighbors_b'],
                                              test_data['pack_neighbors_f'], test_data['label'], test_data['label2'], train=False)
    with tf.variable_scope('model', reuse=True):
        valid_loss, valid_score, _a1,_a2,_a3 = model.forward(valid_data['user'], valid_data['item'], valid_data['biz'],
                                              valid_data['friends'], valid_data['user_items'],
                                              valid_data['user_bizs'],valid_data['user_friends'],
                                              valid_data['user_packages'], valid_data['pack_neighbors_b'],
                                              valid_data['pack_neighbors_f'], valid_data['label'], valid_data['label2'], train=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        step = 0

        for epoch in range(args.epoch):
            sess.run([train_iterator.initializer, test_iterator.initializer])
            t0 = time.time()
            step = 0
            loss = []
            log.print('start training: ')
            score = []
            label = []
            while True:
                try:
                    if step > (n_train*args.ratio)/args.batch_size:
                        break

                    loss_, _, sco, lab,att1,u1,p1 = sess.run([train_loss, train_opt, train_score,train_data['label'],att,u,p])
                    score.extend(sco)
                    label.extend(lab)
                    loss.append(loss_)
                    step += 1
                    if step % 1000 == 0:
                        print(step)
                        tr_auc = roc_auc_score(label, score)
                        print('train auc:%.4f\t' % tr_auc)
                        sess.run(valid_iterator.initializer)
                        _val_loss, auc, f1, acc, prec, rec = eval_epoch(args,sess, valid_score, valid_loss, valid_data,_a1,_a2,_a3)
                        print('---After %d steps' % (step),
                                  'train_loss:%.4f\tvalid_loss:%.4f\tauc:%.4f' % (loss_, _val_loss, auc))
                except tf.errors.OutOfRangeError:
                    break
            t1 = time.time()
            print('finish training: %.4fs'%(t1-t0))
            log.print('start predicting: ')
            _test_loss, auc, f1, acc, prec, rec = eval_epoch(args,sess, test_score, test_loss, test_data,a1,a2,a3,test=True)
            _train_loss = np.mean(np.array(loss))
            t2 = time.time()
            tr_auc = roc_auc_score(label, score)
            print('train auc:%.4f\t' % tr_auc)
            print('Epoch:%d\ttime: %.4fs\ttrain loss:%.4f\ttest loss:%.4f\tauc:%.4f' %
                       (epoch, (t2 - t1), _train_loss, _test_loss, auc))
