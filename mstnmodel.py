"""
Derived from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""
import tensorflow as tf
import numpy as np
import math
import sys
from keras.models import Model,Input
import layers
import costs
sys.path.append('optimizers')
import matplotlib.pyplot as plt
from functools import partial
from tensorflow.python.util import tf_inspect
from tensorflow.python.framework import ops
import utils
from utils import make_layer_list
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
#from AMSGrad import AMSGrad
#ensure the ys and yt are two dimension
from layer import stack_layers
from keras import backend as K
# supervised semantic loss proposed in saied et al. ICCV17
def supervised_semantic_loss(xs,xt,ys,yt):
	#K=int(ys.get_shape()[-1])
	#return tf.constant(0.0)
    K=10
    classloss=tf.constant(0.0)
    for i in range(1,K+1):
        xsi=tf.gather(xs,tf.where(tf.equal(ys,i)))
        xti=tf.gather(xt,tf.where(tf.equal(yt,i)))
        xsi_=tf.expand_dims(xsi,0)
        xti_=tf.expand_dims(xti,1)
        distances=0.5*tf.reduce_sum(tf.squared_difference(xsi_,xti_))
        classloss+=distances
    classloss/=10.0
	
    return 0.0001*classloss
_DIVERGENCES = {}
#squared Euclidean loss for prototypes
def protoloss(sc,tc):
	return tf.reduce_mean((tf.square(sc-tc)))


class LeNetModel(object):

    def __init__(self, num_classes=1000, is_training=True,image_size=28,dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.default_image_size=image_size
        self.is_training=is_training
        self.num_channels=1
        self.mean=None
        self.bgr=False
        self.range=None
        self.featurelen=10
        self.source_moving_centroid=tf.get_variable(name='source_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)
        self.target_moving_centroid=tf.get_variable(name='target_moving_centroid',shape=[num_classes,self.featurelen],initializer=tf.zeros_initializer(),trainable=False)
        self.target_moving_centroid1 = tf.get_variable(name='target_moving_centroid1',shape=[num_classes, self.featurelen],initializer=tf.zeros_initializer(), trainable=False)
        self.target_moving_centroid2 = tf.get_variable(name='target_moving_centroid2',shape=[num_classes, self.featurelen],initializer=tf.zeros_initializer(), trainable=False)
        self.logits_moving_centroid = tf.get_variable(name='logits_moving_centroid',shape=[num_classes, self.featurelen],initializer=tf.zeros_initializer(), trainable=False)


    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 5, 5, 32, 1, 1, padding='VALID',bn=True,name='conv1')
        pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID',name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(pool1, 5, 5, 64, 1, 1, padding='VALID',bn=True,name='conv2')
        pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name ='pool2')

        # conv3 = conv(pool2, 5, 5, 64, 1, 1, padding='VALID',bn=True,name='conv3')
        # pool3 = max_pool(conv3, 2, 2, 2, 2, padding='VALID', name ='pool3')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.contrib.layers.flatten(pool2)
        self.flattened=flattened
        fc1 = fc(flattened, 1024, 500, bn=False,name='fc1')
        fc2 = fc(fc1, 500, 10, relu=False,bn=False,name='fc2')
        self.fc1=fc1
        self.fc2=fc2
        self.score=fc2
        self.output=tf.nn.softmax(self.score)
        self.feature=fc2

        # GAT
        return self.score

    def loss(self, batch_x, batch_y=None):
        with tf.variable_scope('reuse_inference') as scope:
            y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers,global_step):
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print (train_layers)
        X = tf.trainable_variables()
        var_list=[v for v in tf.trainable_variables() if v.name.split('/')[1] in ['conv1','conv2','fc1','fc2','G']]
        self.Gregloss=5e-4*tf.reduce_mean([tf.nn.l2_loss(x) for x in var_list if 'weights' in x.name])
	
        new_weights=[v for v in var_list if 'weights' in v.name or 'kernel' in v.name]
        new_biases=[v for v in var_list if 'biases' in v.name or 'bias' in v.name]
        self.global_step= global_step
	
        print ('==============new_weights=======================')
        print (new_weights)
        print ('==============new_biases=======================')
        print (new_biases)
        self.smloss = self.SemanticlossFG+self.SemanticlossFG1+self.SemanticlossFG2
        self.F_loss=self.Gregloss+self.loss+0.1*self.G_loss+global_step*self.smloss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print ('+++++++++++++++ batch norm update ops +++++++++++++++++')
        print (update_ops)
        with tf.control_dependencies(update_ops):
            train_op3=tf.train.MomentumOptimizer(learning_rate*1.0,0.9).minimize(self.F_loss, var_list=new_weights)
            train_op4=tf.train.MomentumOptimizer(learning_rate*2.0,0.9).minimize(self.F_loss, var_list=new_biases)
        train_op=tf.group(train_op3,train_op4)

        return train_op
    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('bvlc_alexnet.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope('reuse_inference/'+op_name, reuse=True):
                print ('=============================OP_NAME  ========================================')
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
                        print (op_name,var)
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
                        print (op_name,var)
                        session.run(var.assign(data))
    def gatfun(self,inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        with tf.variable_scope('res/G'):
            attns = []
            for _ in range(n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    out_sz=hid_units[0], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=True))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                        out_sz=hid_units[i], activation=activation,
                        in_drop=ffd_drop, coef_drop=attn_drop, residual=True))
                h_1 = tf.concat(attns, axis=-1)
            out = []
            for i in range(n_heads[-1]):
                out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=nb_classes, activation=lambda x: x,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=True))
            logits = tf.add_n(out) / n_heads[-1]
            return logits

    def adoptimize(self, learning_rate, train_layers=[]):
        var_list = [v for v in tf.trainable_variables() if 'D' in v.name]
        D_weights = [v for v in var_list if 'weights' in v.name]
        D_biases = [v for v in var_list if 'biases' in v.name]
        print('=================Discriminator_weights=====================')
        print(D_weights)
        print('=================Discriminator_biases=====================')
        print(D_biases)

        self.Dregloss = 5e-4 * tf.reduce_mean([tf.nn.l2_loss(v) for v in var_list if 'weights' in v.name])
        D_op1 = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.D_loss + self.Dregloss, var_list=D_weights)
        D_op2 = tf.train.MomentumOptimizer(learning_rate * 2.0, 0.9).minimize(self.D_loss + self.Dregloss,var_list=D_biases)

        D_op = tf.group(D_op1, D_op2)
        return D_op


    def tcloss(self,x,xt,y,yt,xt1,yt1,xt2,yt2,y_mask,yt_mask):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            source_feature = self.inference(x,True)
            source_fc1 = self.fc1
            scope.reuse_variables()
            target_feature = self.inference(xt,True)
            target_fc1 = self.fc1
            self.tc=self.score
            scope.reuse_variables()
            target_feature1 = self.inference(xt1,True)
            target1_fc1 = self.fc1
            self.tc1 = self.score
            scope.reuse_variables()
            target_feature2 = self.inference(xt2,True)
            target2_fc1 = self.fc1
            self.tc2 = self.score
        input_affinity = tf.concat([source_fc1, target_fc1,target1_fc1,target2_fc1], axis=0)
        nb_nodes = input_affinity.shape[0]

        features = input_affinity[np.newaxis]
        scale_nbr = 2
        hid_units = [50]
        n_heads = [8, 4]
        residual = False


        n_nbrs = 3
        W11 = costs.knn_affinity(source_feature, source_feature, n_nbrs, scale_nbr=scale_nbr)
        W12 = costs.knn_affinity(source_feature, target_feature, n_nbrs, scale_nbr=scale_nbr)
        W13 = costs.knn_affinity(source_feature, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W14 = costs.knn_affinity(source_feature, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W22 = costs.knn_affinity(target_feature, target_feature, n_nbrs, scale_nbr=scale_nbr)
        W23 = costs.knn_affinity(target_feature, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W24 = costs.knn_affinity(target_feature, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W33 = costs.knn_affinity(target_feature1, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W34 = costs.knn_affinity(target_feature1, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W44 = costs.knn_affinity(target_feature2, target_feature2, n_nbrs, scale_nbr=scale_nbr)

                # #
        W_1 = tf.concat([W11, W12, W13,W14], 1)
        W_2 = tf.concat([tf.transpose(W12), W22, W23, W24], 1)
        W_3 = tf.concat([tf.transpose(W13), tf.transpose(W23), W33, W34], 1)
        W_4 = tf.concat([tf.transpose(W14), tf.transpose(W24), tf.transpose(W34), W44], 1)
        W1 = tf.concat([W_1, W_2, W_3,W_4], 0)
        W = tf.ceil(W1, name=None)



        W = -1e9 * (1.0 - W)+1*W1
        adj = W[np.newaxis]




        with tf.variable_scope(tf.get_variable_scope(), reuse=True):

            logits = self.gatfun(features,self.num_classes, nb_nodes, False, 0.0, 0.0,
                                    bias_mat=adj,
                                    hid_units=hid_units, n_heads=n_heads,
                                    activation=tf.nn.elu, residual=residual)




        c = tf.split(logits[0], 4, 0)

        source_result = tf.argmax(y, 1)
        target_result = tf.argmax(tf.nn.softmax(c[1]), 1)
        target_result1 = tf.argmax(tf.nn.softmax(c[2]), 1)
        target_result2 = tf.argmax(tf.nn.softmax(c[3]), 1)



        ones = tf.ones_like(source_feature)
        current_source_count = tf.unsorted_segment_sum(ones, source_result, self.num_classes)
        ones = tf.ones_like(target_feature)
        current_target_count = tf.unsorted_segment_sum(ones, target_result, self.num_classes)
        ones = tf.ones_like(target_feature1)
        current_target_count1 = tf.unsorted_segment_sum(ones, target_result1, self.num_classes)
        ones = tf.ones_like(target_feature2)
        current_target_count2 = tf.unsorted_segment_sum(ones, target_result2, self.num_classes)


        current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))
        current_positive_target_count = tf.maximum(current_target_count, tf.ones_like(current_target_count))
        current_positive_target_count1 = tf.maximum(current_target_count1, tf.ones_like(current_target_count1))
        current_positive_target_count2 = tf.maximum(current_target_count2, tf.ones_like(current_target_count2))
        current_source_centroid = tf.divide(tf.unsorted_segment_sum(data=source_feature, segment_ids=source_result, num_segments=self.num_classes),current_positive_source_count)
        current_target_centroid = tf.divide(tf.unsorted_segment_sum(data=target_feature, segment_ids=target_result, num_segments=self.num_classes),current_positive_target_count)
        current_target_centroid1 = tf.divide(tf.unsorted_segment_sum(data=target_feature1, segment_ids=target_result1, num_segments=self.num_classes),current_positive_target_count1)
        current_target_centroid2 = tf.divide(tf.unsorted_segment_sum(data=target_feature2, segment_ids=target_result2, num_segments=self.num_classes),current_positive_target_count2)

        source_decay = tf.constant(.3)
        target_decay = tf.constant(.3)


        source_centroid = (source_decay) * current_source_centroid + ( 1. - source_decay) * self.source_moving_centroid
        target_centroid = (target_decay) * current_target_centroid + ( 1. - target_decay) * self.target_moving_centroid
        target_centroid1 = (target_decay) * current_target_centroid1 + (1. - target_decay) * self.target_moving_centroid1
        target_centroid2 = (target_decay) * current_target_centroid2 + (1. - target_decay) * self.target_moving_centroid2
        self.SemanticlossFG = protoloss(source_centroid, target_centroid)
        self.SemanticlossFG1 = protoloss(source_centroid, target_centroid1)
        self.SemanticlossFG2= protoloss(source_centroid, target_centroid2)

        cnn_feature = tf.concat([source_feature, target_feature, target_feature1, target_feature2], axis=0)

        with tf.variable_scope('reuse') as scope:
            source_logits,_=D(cnn_feature)
            scope.reuse_variables()
            target_logits,_=D(logits[0])


        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logits, labels=tf.zeros_like(source_logits)))
        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits, labels=tf.ones_like(target_logits)))


        self.D_loss=0.1*(1*D_fake_loss+D_real_loss)
        self.G_loss = -self.D_loss


        self.JSD = -(3*D_fake_loss+D_real_loss)/2+3*math.log(2)




        return self.JSD

    def masked_accuracy(self,logits, labels, y_mask,yt_mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask=tf.concat([y_mask, yt_mask], axis=0)
        mask = tf.cast(mask, dtype=tf.float32)
        # mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def graph_loss(self,x,xt,y,yt,xt1,yt1,xt2,yt2,y_mask,yt_mask):
        with tf.variable_scope('reuse_inference') as scope:
            scope.reuse_variables()
            source_feature = self.inference(x)
            source_fc1 = self.fc1
            scope.reuse_variables()
            target_feature = self.inference(xt)
            target_fc1 = self.fc1
            scope.reuse_variables()
            target_feature1 = self.inference(xt1)
            target1_fc1 = self.fc1
            scope.reuse_variables()
            target_feature2 = self.inference(xt2)
            target2_fc1 = self.fc1
        input_affinity = tf.concat([source_fc1, target_fc1, target1_fc1,target2_fc1], axis=0)
        y_all = tf.concat([y, yt, yt1,yt2], axis=0)
        Y1_mask = tf.concat([y_mask, yt_mask,yt_mask,yt_mask], axis=0)
        Y1_mask = tf.transpose(Y1_mask)

        nb_nodes = input_affinity.shape[0]
        attn_drop = 0.3
        ffd_drop =0.3
        scale_nbr = 2
        hid_units = [50]  # numbers of hidden units per each attention head in each layer
        n_heads = [8,4]  # additional entry for the output layer
        residual = False
        #W
        n_nbrs = 3


        features = input_affinity[np.newaxis]
        W11 = costs.knn_affinity(source_feature, source_feature, n_nbrs, scale_nbr=scale_nbr)
        W12 = costs.knn_affinity(source_feature, target_feature, n_nbrs, scale_nbr=scale_nbr)
        W13 = costs.knn_affinity(source_feature, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W14 = costs.knn_affinity(source_feature, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W22 = costs.knn_affinity(target_feature, target_feature, n_nbrs, scale_nbr=scale_nbr)
        W23 = costs.knn_affinity(target_feature, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W24 = costs.knn_affinity(target_feature, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W33 = costs.knn_affinity(target_feature1, target_feature1, n_nbrs, scale_nbr=scale_nbr)
        W34 = costs.knn_affinity(target_feature1, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        W44 = costs.knn_affinity(target_feature2, target_feature2, n_nbrs, scale_nbr=scale_nbr)

        # #
        W_1 = tf.concat([W11, W12, W13,W14], 1)
        W_2 = tf.concat([tf.transpose(W12), W22, W23, W24], 1)
        W_3 = tf.concat([tf.transpose(W13), tf.transpose(W23), W33, W34], 1)
        W_4 = tf.concat([tf.transpose(W14), tf.transpose(W24), tf.transpose(W34), W44], 1)
        W1 = tf.concat([W_1, W_2, W_3,W_4], 0)
        W = tf.ceil(W1, name=None)
        W = -1e9 * (1.0 - W)+1*W1
        adj = W[np.newaxis]
        logits = self.gatfun(features, self.num_classes, nb_nodes, True, attn_drop, ffd_drop,
                                 bias_mat=adj,
                                 hid_units=hid_units, n_heads=n_heads,
                                 activation=tf.nn.elu, residual=residual)
        self.loss_graph = tf.nn.softmax_cross_entropy_with_logits(logits=logits[0], labels=y_all)
        self.loss_graph = tf.multiply(self.loss_graph,Y1_mask)
        self.loss_graph = 4*tf.reduce_mean(self.loss_graph)



        c = tf.split(logits[0], 4, 0)
        source_result = tf.argmax(y, 1)

        target_feature = tf.concat([c[1], c[2], c[3]], axis=0)
        target_result = tf.nn.softmax(target_feature)
        target_result = tf.argmax(target_result, 1)
        #

        ones = tf.ones_like(source_feature)
        current_logits_count = tf.unsorted_segment_sum(ones, source_result, self.num_classes)
        current_positive_logits_count = tf.maximum(current_logits_count, tf.ones_like(current_logits_count))
        current_logits_centroid = tf.divide(
            tf.unsorted_segment_sum(data=source_feature, segment_ids=source_result, num_segments=self.num_classes),
            current_positive_logits_count)

        ones = tf.ones_like(target_feature)
        current_source_count = tf.unsorted_segment_sum(ones, target_result, self.num_classes)
        current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))
        current_source_centroid = tf.divide(
            tf.unsorted_segment_sum(data=target_feature, segment_ids=target_result, num_segments=self.num_classes),
            current_positive_source_count)

        source_decay = tf.constant(.3)
        logits_decay = tf.constant(.3)
        source_centroid = (source_decay) * current_source_centroid + (1. - source_decay) * self.source_moving_centroid
        logits_centroid = (logits_decay) * current_logits_centroid + (1. - logits_decay) * self.logits_moving_centroid



        self.Semanticloss_G = 1 * protoloss(source_centroid, logits_centroid)

        return self.loss_graph

    def graph_optimize(self, lr, l2_coef,global_step):

        var_list = [v for v in tf.trainable_variables() if 'G' in v.name]
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list if v.name not
                           in ['biases', 'gamma', 'kernel', 'beta']]) * l2_coef


        new_weights = [v for v in var_list if 'kernel' in v.name or 'gamma' in v.name]
        new_biases = [v for v in var_list if 'biases' in v.name or 'beta' in v.name]

        print ('==============new_weights=======================')
        print (new_weights)
        print ('==============new_biases=======================')
        print (new_biases)


        train_op3 = tf.train.AdamOptimizer(lr * 1.0).minimize(self.loss_graph+lossL2+0*self.Semanticloss_G, var_list=new_weights)
        train_op4 = tf.train.AdamOptimizer(lr * 2.0).minimize(self.loss_graph+lossL2+0*self.Semanticloss_G, var_list=new_biases)
        train_op1 = tf.group(train_op3,train_op4)



        return train_op1









"""
Helper methods
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, bn=False,padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels/groups, num_filters],initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
        if bn==True:
            bias=tf.contrib.layers.batch_norm(bias,scale=True)
        relu = tf.nn.relu(bias, name=scope.name)
        return relu








def fc(x, num_in, num_out, name, relu=True,bn=False,stddev=0.001):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in,num_out],initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if bn==True:
            act=tf.contrib.layers.batch_norm(act,scale=True)
        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def outer(a,b):
        a=tf.reshape(a,[-1,a.get_shape()[-1],1])
        b=tf.reshape(b,[-1,1,b.get_shape()[-1]])
        c=a*b
        return tf.contrib.layers.flatten(c)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1],
                          padding = padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def D(x):
    with tf.variable_scope('D'):
        num_units_in = int(x.get_shape()[-1])
        num_units_out = 1
        n = 500
        weights = tf.get_variable('weights', shape=[num_units_in, n],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', shape=[n], initializer=tf.zeros_initializer())
        hx = (tf.matmul(x, weights) + biases)
        ax = tf.nn.relu(hx)

        weights2 = tf.get_variable('weights2', shape=[n, n], initializer=tf.contrib.layers.xavier_initializer())
        biases2 = tf.get_variable('biases2', shape=[n], initializer=tf.zeros_initializer())
        hx2 = (tf.matmul(ax, weights2) + biases2)
        ax2 = tf.nn.relu(hx2)
        weights3 = tf.get_variable('weights3', shape=[n, num_units_out],
                                   initializer=tf.contrib.layers.xavier_initializer())
        biases3 = tf.get_variable('biases3', shape=[num_units_out], initializer=tf.zeros_initializer())
        hx3 = tf.matmul(ax2, weights3) + biases3
        return hx3, tf.nn.sigmoid(hx3)



