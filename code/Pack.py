# -*- coding: utf-8 -*-
# @Create Time : 2020/7/13 15:10
# @Author : lee
# @FileName : Pack.py
import tensorflow as tf
import math
class Model():
    def __init__(self, args, n_users, n_items, n_bizs, f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f, u_max_i, u_max_f):

        self.batch_size = args.batch_size
        self.K = args.K
        self.lr = args.lr
        self.reg = args.reg
        self.n_users = n_users
        self.n_items = n_items
        self.n_bizs = n_bizs
        self.emb_dim = args.dimension
        self.f_max_len, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f = f_max_len, u_max_pack, pack_max_nei_b, pack_max_nei_f
        self.u_max_i, self.u_max_f = u_max_i, u_max_f
        self.params = []
        self.stdv = 1.0 / math.sqrt(self.emb_dim)
        self.keep_prob = 1 - args.drop_out

        self.user_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_users, self.emb_dim],
                                              dtype=tf.float32, name='user_embedding')
        self.user_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.user_embedding], 0)
        self.item_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_items, self.emb_dim],
                                              dtype=tf.float32, name='item_embedding')
        self.item_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.item_embedding], 0)
        self.biz_embedding = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[self.n_bizs, self.emb_dim],
                                              dtype=tf.float32, name='biz_embedding')
        self.biz_embedding = tf.concat([tf.zeros(shape=[1, self.emb_dim]), self.biz_embedding], 0)
        self.params.append(self.user_embedding)
        self.params.append(self.item_embedding)
        self.params.append(self.biz_embedding)
        
        self.weight_size = [64,32]
        self.n_layers = len(self.weight_size)
        self.weight_size_list = [3 * self.emb_dim] + self.weight_size
        self.weights = {}
        initializer = tf.contrib.layers.xavier_initializer()
        for i in range(self.n_layers):
            self.weights['W_%d' %i] = tf.Variable(
                initializer([self.weight_size_list[i], self.weight_size_list[i+1]]), name='W_%d' %i)
            self.weights['b_%d' %i] = tf.Variable(
                initializer([1, self.weight_size_list[i+1]]), name='b_%d' %i)

        self.weights['h'] = tf.Variable(initializer([self.weight_size_list[-1], 1]), name='h')

    def forward(self, user, item, biz, friends, user_items, user_bizs, user_friends, user_packages,
                pack_neighbors_b, pack_neighbors_f, label, label2, train):
        if not train:
            self.keep_prob = 1

        user_emb = tf.nn.embedding_lookup(self.user_embedding, user) # B*D
       
        up_mask = tf.expand_dims(tf.cast(tf.sign(tf.abs(tf.reduce_sum(user_packages, axis=-1))), dtype=tf.float32),-1) # B*N*1
        pb_mask = tf.expand_dims(tf.cast(tf.sign(tf.abs(tf.reduce_sum(pack_neighbors_b, axis=-1))), dtype=tf.float32),-1)
        pf_mask = tf.expand_dims(tf.cast(tf.sign(tf.abs(tf.reduce_sum(pack_neighbors_f, axis=-1))), dtype=tf.float32),-1)

        [up_items, up_bizs, up_friends] = tf.split(user_packages, [1, 1, self.f_max_len], axis=-1)  # B * u_max_p * 1, B * u_max_p * 1, B * u_max_p * max_f
        [pb_items, pb_bizs, pb_friends] = tf.split(pack_neighbors_b, [1, 1, self.f_max_len], axis=-1) # B * p_max_nei(biz) * 1
        [pf_items, pf_bizs, pf_friends] = tf.split(pack_neighbors_f, [1, 1, self.f_max_len], axis=-1) # B * p_max_nei(fri) * 1
        _items = tf.concat([tf.reshape(item, [-1, 1, 1]), up_items, pb_items, pf_items], axis=1)
        _bizs = tf.concat([tf.reshape(biz, [-1, 1, 1]), up_bizs, pb_bizs, pf_bizs], axis=1)
        _friends = tf.concat([tf.expand_dims(friends, axis=1), up_friends, pb_friends, pf_friends], axis=1)

        
        
        user_emb,a2,a3,ta = self.dual_aggregate(user_emb, user_items, user_bizs, user_friends,train)
        # #user_emb = user_emb + tf.reduce_mean(u_packs*up_mask, axis=1)
        
        
        intra_packages,att = self.intra(user_emb, tf.reshape(_friends,[-1, self.f_max_len]), tf.reshape(_items,[-1]), tf.reshape(_bizs,[-1]), train)
        print(intra_packages,att)
        intra_packages = tf.reshape(intra_packages, [self.batch_size, -1, self.emb_dim])
        att = tf.reshape(att, [self.batch_size, -1, 7])
        [tar_pack, u_packs, pb_packs, pf_packs] = tf.split(intra_packages, [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f], axis=1)
        [tar_att, u_att, pb_att, pf_att] = tf.split(att, [1, self.u_max_pack, self.pack_max_nei_b, self.pack_max_nei_f], axis=1)
        tar_pack = tf.reshape(tar_pack, [self.batch_size, self.emb_dim])
        tar_att = tf.reshape(tar_att, [self.batch_size, 7])
        u_packs = tf.reshape(u_packs, [self.batch_size, -1, self.emb_dim])
        pb_packs = tf.reshape(pb_packs, [self.batch_size, -1, self.emb_dim])
        pf_packs = tf.reshape(pf_packs, [self.batch_size, -1, self.emb_dim])

        # tar_pack = tf.reshape(intra_packages, [self.batch_size, self.emb_dim])
        
        # a = tf.reduce_mean(pb_packs*pb_mask, axis=1)
        # pack_emb = tar_pack + tf.reduce_mean(pb_packs*pb_mask, axis=1) + tf.reduce_mean(pf_packs*pf_mask, axis=1)
        
        # gate_attention
        pack_emb = tar_pack + tf.reduce_mean(self.gate_attention(tar_pack,pb_packs,pb_mask,self.emb_dim,'biz'), axis=1) \
                   + tf.reduce_mean(self.gate_attention(tar_pack,pf_packs,pf_mask,self.emb_dim,'friend'), axis=1)
        
        # gate_attention
        user_emb = user_emb + tf.reduce_mean(self.gate_attention(user_emb,u_packs,up_mask,self.emb_dim,'user'), axis=1)
        
        # pack_emb = tar_pack
        item_emb = tf.nn.embedding_lookup(self.item_embedding, item)
        
        
        z = []
        z.append(tf.concat([user_emb, pack_emb, user_emb * pack_emb], 1))
        

        for i in range(self.n_layers):
            temp = tf.nn.relu(tf.matmul(z[i], self.weights['W_%d' % i]) + self.weights['b_%d' % i])
            temp = tf.nn.dropout(temp, self.keep_prob)
            z.append(temp)

        agg_out = tf.matmul(z[-1], self.weights['h'])
        
        self.scores = tf.squeeze(agg_out)
        self.scores_normalized = tf.sigmoid(self.scores)
        # self.predict_label = tf.cast(self.scores > 0.5, tf.int32)

        with tf.variable_scope('train'):

            
            base_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(label,tf.float32), logits=self.scores))
                
                
            l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
            for param in self.params:
                l2_loss = tf.add(l2_loss, self.reg * tf.nn.l2_loss(param))

            loss = base_loss + l2_loss 
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)
            # optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
            # optimizer = tf.train.AdagradOptimizer(self.lr).minimize(loss)

        if train:
            return loss, self.scores_normalized, optimizer, tar_att, user_emb, user_emb, user_emb
        else:
            return loss, self.scores_normalized, tar_att, user_emb, user_emb#a2,a3,ta
    

    def intra(self, user_emb, friends, item, biz, train):
        _user_emb = tf.reshape(tf.tile(tf.expand_dims(user_emb, axis=1),
                                      [1, 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f, 1]),
                              [-1, self.emb_dim, 1])  # BN*D*1
        friend_emb = tf.nn.embedding_lookup(self.user_embedding, friends)  # BN*F*D, N max neighbor size, M max friend size.
        masks = tf.sign(tf.abs(tf.reduce_sum(friend_emb, axis=-1)))  # BN*F
        item_emb = tf.nn.embedding_lookup(self.item_embedding, item)  # BN*D
        biz_emb = tf.nn.embedding_lookup(self.biz_embedding, biz)  # BN*D
        

        # social influence
        f_list = []
        for i in range(self.K):
            w_k = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                self.emb_dim, self.emb_dim], dtype=tf.float32, name='wk_%d' % i)
            w_i = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                self.emb_dim , self.emb_dim], dtype=tf.float32, name='wi_%d' % i)

            # BN*F*D dot D*D -> BN*F*D
            f_k_emb = tf.tensordot(friend_emb, w_k, axes=1)  # BN*F*D
            _item = tf.expand_dims(tf.matmul(item_emb, w_i),1) # BN*1*D
            
            inputs = tf.concat([tf.tile(_item,[1,self.f_max_len,1]),f_k_emb],-1)
            w_omega = tf.get_variable('w_omega_%d'%i, initializer=tf.random_normal([2*self.emb_dim, 1], stddev=0.1))
            b_omega = tf.get_variable('b_omega_%d'%i, initializer=tf.random_normal([1], stddev=0.1))
            u_omega = tf.get_variable('o_omega_%d'%i, initializer=tf.random_normal([1], stddev=0.1))
            self.params.extend([w_k,w_i,w_omega])
            
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            
            vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
            paddings = tf.ones_like(vu) * (-2 ** 32 + 1)
            x = tf.where(tf.equal(masks, 0), paddings, vu)
            att = tf.nn.softmax(x, axis=-1)
            output = tf.reduce_sum(f_k_emb * tf.expand_dims(att, -1), 1, keep_dims=True) # BN*1*D
            
            f_list.append(output)  # K*BN*D
        
        f_K_emb = tf.concat(f_list,1)
        
        t_user = tf.reshape(tf.tile(tf.expand_dims(user_emb, axis=1),
                                      [1, 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f, 1]),
                              [-1, 1, self.emb_dim])
        inputs = tf.concat([tf.tile(t_user,[1,self.K,1]),f_K_emb],-1)
        
        w_a = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(), shape=[
                2*self.emb_dim , self.emb_dim], dtype=tf.float32, name='w_a_d')
        inputs = tf.nn.relu(tf.tensordot(inputs,w_a, axes=1))
        inputs = tf.nn.dropout(inputs, self.keep_prob)
        w_omega = tf.get_variable('w_omega_d', initializer=tf.random_normal([self.emb_dim, 1], stddev=0.1))
        b_omega = tf.get_variable('b_omega_d', initializer=tf.random_normal([1], stddev=0.1))
        u_omega = tf.get_variable('o_omega_d', initializer=tf.random_normal([1], stddev=0.1))
        self.params.extend([w_omega])
        
        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        
        vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
        
        att = tf.nn.softmax(vu, axis=-1)
        f_emb = tf.reduce_sum(f_K_emb * tf.expand_dims(att, -1), 1) # BN*D
        
        
        
        
        
        
        
        
        
        
        
        
        # interaction
        pack = [f_emb, item_emb, biz_emb, f_emb*item_emb, f_emb*biz_emb, item_emb*biz_emb, f_emb*item_emb*biz_emb]
        # pack = [item_emb, biz_emb, item_emb*biz_emb]
        pack = tf.transpose(pack, perm=[1, 0, 2])
        # user_emb B*D  pack BN*7*D
        # pack_emb, att,_,__ = self.attention(tf.transpose(_user_emb, perm=[0,2,1]), pack, 7, self.batch_size * (
                    # 1 + self.u_max_pack + self.pack_max_nei_b + self.pack_max_nei_f))
        # pack_emb = tf.reduce_mean(pack,1)
        masks = tf.sign(tf.abs(tf.reduce_sum(pack, axis=-1)))
        _user_emb = tf.tile(tf.transpose(_user_emb, perm=[0,2,1]),[1,7,1])
        pack_emb, att = self._attention(_user_emb, pack, 2*self.emb_dim, self.emb_dim,'pack_attention',train, masks)
        
        return pack_emb,att
    #
    #
    # def inter(self, c_pack, n_packs):

    def dual_aggregate(self, user_emb, items, bizs, friends,train):
        friends_emb = tf.nn.embedding_lookup(self.user_embedding, friends) #B*M*D
        items_emb = tf.nn.embedding_lookup(self.item_embedding, items)  # B*N*D
        bizs_emb = tf.nn.embedding_lookup(self.biz_embedding, bizs)
        # self.params.append(user_emb)
        user_emb_ = tf.expand_dims(user_emb, axis=1)

        with tf.variable_scope("friends", reuse=tf.AUTO_REUSE):
            # friend_type, att1,m1,x1 = self.attention(user_emb_, friends_emb, self.u_max_pack*self.f_max_len, self.batch_size)
            f_masks = tf.sign(tf.abs(tf.reduce_sum(friends_emb, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,self.u_max_f,1])
            friend_type, att1 = self._attention(_user_emb,friends_emb, 2*self.emb_dim, self.emb_dim, 'friends', train,f_masks)
            
            ### self connection
            # w = tf.get_variable('wf', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # friend_type = tf.nn.relu(tf.matmul(tf.concat([friend_type, user_emb],-1), w))

        with tf.variable_scope("items", reuse=tf.AUTO_REUSE):
            # item_type, att2,m2,x2 = self.attention(user_emb_, items_emb, self.u_max_pack, self.batch_size)
            i_masks = tf.sign(tf.abs(tf.reduce_sum(items_emb, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,self.u_max_i,1])
            ### inputs = tf.concat([_user_emb, items_emb], -1)
            item_type, att2 = self._attention(_user_emb,items_emb, 2*self.emb_dim, self.emb_dim,'items', train,i_masks)
            
            # w = tf.get_variable('wi', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # item_type = tf.nn.relu(tf.matmul(tf.concat([item_type, user_emb],-1), w))
            
        with tf.variable_scope("bizs", reuse=tf.AUTO_REUSE):
            # biz_type, att3,m,x = self.attention(user_emb_, bizs_emb, self.u_max_pack, self.batch_size)  # B*D
            b_masks = tf.sign(tf.abs(tf.reduce_sum(bizs_emb, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,self.u_max_i,1])
            ### inputs = tf.concat([_user_emb, bizs_emb], -1)
            biz_type, att2 = self._attention(_user_emb, bizs_emb, 2*self.emb_dim, self.emb_dim,'bizs', train, b_masks)
            
            # w = tf.get_variable('wbi', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            # biz_type = tf.nn.relu(tf.matmul(tf.concat([biz_type, user_emb],-1), w))
            
        with tf.variable_scope("type_attention", reuse=tf.AUTO_REUSE):
            # n_emb = tf.concat([tf.expand_dims(friend_type, axis=1), tf.expand_dims(item_type, axis=1),
                               # tf.expand_dims(biz_type, axis=1)], axis=1)
            # _user_emb, t_att,mm,xx = self.attention(user_emb_, n_emb, 3, self.batch_size)
            
            inputs = tf.concat([tf.expand_dims(friend_type, axis=1), tf.expand_dims(item_type, axis=1),
                               tf.expand_dims(biz_type, axis=1)], axis=1)
            masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
            _user_emb = tf.tile(user_emb_,[1,3,1])
            #### inputs = tf.concat([_user_emb, inputs], -1)
            _user_emb, t_att = self._attention(_user_emb, inputs, 2*self.emb_dim, self.emb_dim,'type_attention',train, masks)
            
            #### self connection
            w = tf.get_variable('w_self', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
            user_emb = tf.nn.relu(tf.matmul(tf.concat([_user_emb, user_emb],-1), w))
        
        
        # user_emb = tf.reduce_mean(tf.concat([tf.reduce_mean(friends_emb,axis=1,keep_dims=True),tf.reduce_mean(items_emb,axis=1,keep_dims=True),tf.reduce_mean(bizs_emb,axis=1,keep_dims=True)],axis=1),axis=1)
        return user_emb, att1,att1,att1


    def attention(self, user_emb, node_emb, n_nodes, batch_size):
        user_emb = tf.tile(user_emb, [1, n_nodes, 1])
        masks = tf.sign(tf.abs(tf.reduce_sum(node_emb, axis=-1)))  # B*N
        # masks = tf.expand_dims(masks, axis=-1)
        w1 = tf.get_variable('w1', initializer=tf.contrib.layers.xavier_initializer(), shape=[2*self.emb_dim, self.emb_dim])
        b1 = tf.get_variable('b1', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.emb_dim])
        w2 = tf.get_variable('w2', initializer=tf.contrib.layers.xavier_initializer(), shape=[self.emb_dim, 1])
        b2 = tf.get_variable('b2', initializer=tf.contrib.layers.xavier_initializer(), shape=[1])
        self.params.extend([w1,w2])
        # print("-----\n",w1)
        x = tf.reshape(tf.concat([user_emb, node_emb], axis=-1), [-1, 2*self.emb_dim])
        x = tf.nn.relu(tf.matmul(x, w1)+b1)
        x = tf.nn.dropout(x, self.keep_prob)
        x = tf.matmul(x, w2)  # BN*1
        # x = tf.nn.dropout(x, self.keep_prob)
        x = tf.reshape(x, [-1,n_nodes])
        paddings = tf.ones_like(x) * (-2 ** 32 + 1)
        x = tf.where(tf.equal(masks, 0), paddings, x)
        att = tf.nn.softmax(x, axis=-1)
        return tf.reduce_sum(tf.expand_dims(att,-1) * node_emb, axis=1), att,masks,x
    
    def _attention(self, inputs1,inputs2, emb_dim1, emb_dim2, name_scope, train, masks = None):
        with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):
            w_omega = tf.get_variable('w_omega', initializer=tf.random_normal([emb_dim2, 1], stddev=0.1))
            w = tf.get_variable('w_t', initializer=tf.random_normal([emb_dim1, emb_dim2], stddev=0.1))
            b_omega = tf.get_variable('b_omega', initializer=tf.random_normal([1], stddev=0.1))
            u_omega = tf.get_variable('o_omega', initializer=tf.random_normal([1], stddev=0.1))
            self.params.extend([w_omega,w])
            inputs = tf.concat([inputs1,inputs2],-1)
            # inputs = tf.nn.relu(tf.tensordot(inputs,w, axes=1))
            inputs = tf.nn.relu(tf.layers.batch_normalization(tf.tensordot(inputs,w, axes=1),training=train))
            inputs = tf.nn.dropout(inputs, self.keep_prob)
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (BN,F,D)*(D,A)=(BN,F,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            
            vu = tf.tensordot(v, u_omega, axes=1, name='vu') # BN*F
            
            paddings = tf.ones_like(vu) * (-2 ** 32 + 1)
            vu = tf.where(tf.equal(masks, 0), paddings, vu)
            att = tf.nn.softmax(vu, axis=-1)
            f_emb = tf.reduce_sum(inputs2 * tf.expand_dims(att, -1), 1) # BN*D
            return f_emb, att
            
    def gate_attention(self, input1, input2, mask, emb_dim, name_scope):
        with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):
            w_g1 = tf.get_variable('w_gate1', initializer=tf.random_normal([emb_dim, emb_dim], stddev=0.1))
            w_g2 = tf.get_variable('w_gate2', initializer=tf.random_normal([emb_dim, emb_dim], stddev=0.1))
            b = tf.get_variable('b_gate', initializer=tf.random_normal([emb_dim], stddev=0.1))
            att = tf.nn.sigmoid(tf.expand_dims(tf.matmul(input1, w_g1),1) + tf.tensordot(input2, w_g2,axes=1) + b)
            att = att * mask
            return att * input2