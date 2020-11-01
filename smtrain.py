import os, sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from mstnmodel import LeNetModel
from mnist import MNIST
from svhn import SVHN
from usps import USPS
from mnistm import MNISTM
from tensorflow.python import debug as tf_debug
from preprocessing import preprocessing
from sklearn import manifold
import math
from tensorflow.contrib.tensorboard.plugins import projector

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 100000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '256,257', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at networs input size')
tf.app.flags.DEFINE_string('train_root_dir', '../training', 'Root directory to put the training data')
tf.app.flags.DEFINE_integer('log_step', 10000, 'Logging period in terms of iteration')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
NUM_CLASSES = 10
import scipy.io as scio

TRAIN_FILE='svhn'
TEST_FILE='mnist,usps,mnistm'
print (TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE)
print (TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE)
print (TRAIN_FILE+'  --------------------------------------->   '+TEST_FILE)
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
TRAIN = MNISTM('data/mnist', split='train', shuffle=True)
VALID = MNIST('data/mnist', split='train', shuffle=True)
VALID1 = USPS('data/usps', split='train', shuffle=True)
VALID2=SVHN('data/svhn',split='train',shuffle=True)

TEST = MNIST('data/mnist', split='test', shuffle=False)
TEST1 = USPS('data/usps', split='test', shuffle=False)
TEST2 = SVHN('data/svhn',split='test',shuffle=False)

FLAGS = tf.app.flags.FLAGS
MAX_STEP = 10000


def decay(start_rate,epoch,num_epochs):
        return start_rate/pow(1+0.001*epoch,0.75)

def adaptation_factor(x):
	den=1.0+math.exp(-10*x)
	lamb=2.0/den-1.0
	return min(lamb,1.0)
def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('alexnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.train_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.train_root_dir): os.mkdir(FLAGS.train_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('dropout_keep_prob={}\n'.format(FLAGS.dropout_keep_prob))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('train_root_dir={}\n'.format(FLAGS.train_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()
    
    adlamb=tf.placeholder(tf.float32,name='adlamb')
    num_update=tf.placeholder(tf.float32,name='num_update')
    decay_learning_rate=tf.placeholder(tf.float32)
    dropout_keep_prob = tf.placeholder(tf.float32)
    is_training=tf.placeholder(tf.bool)    
    time=tf.placeholder(tf.float32,[1])

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = LeNetModel(num_classes=NUM_CLASSES, image_size=28,is_training=is_training,dropout_keep_prob=dropout_keep_prob)
    # Placeholders
    x_s = tf.placeholder(tf.float32, [None, 28, 28, 3],name='x')
    x_t = tf.placeholder(tf.float32, [None, 28, 28, 1],name='xt')
    x_t1= tf.placeholder(tf.float32, [None, 16, 16, 1],name='xt1')
    x_t2 = tf.placeholder(tf.float32, [None, 32, 32, 3], name='xt2')
    x=preprocessing(x_s,model)
    xt=preprocessing(x_t,model)
    xt1 = preprocessing(x_t1, model)
    xt2 = preprocessing(x_t2, model)

    y = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='y')
    yt = tf.placeholder(tf.float32, [None, NUM_CLASSES],name='yt')
    yt1 = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='yt1')
    yt2 = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='yt2')
    y_mask = tf.placeholder(tf.float32, [None, 1],name='y_mask')
    yt_mask = tf.placeholder(tf.float32, [None, 1],name='yt_mask')
    loss = model.loss(x, y)

    t_s = model.score
    # Training accuracy of the model
    source_correct_pred = tf.equal(tf.argmax(model.score, 1), tf.argmax(y, 1))
    source_correct=tf.reduce_sum(tf.cast(source_correct_pred,tf.float32))
    source_accuracy = tf.reduce_mean(tf.cast(source_correct_pred, tf.float32))



    loss_graph = model.graph_loss(x,xt,y,yt,xt1,yt1,xt2,yt2,y_mask,yt_mask)
    tcloss= model.tcloss(x,xt,y,yt,xt1,yt1,xt2,yt2,y_mask,yt_mask)
    correct_pred = tf.equal(tf.argmax(model.tc, 1), tf.argmax(yt, 1))
    correct=tf.reduce_sum(tf.cast(correct_pred,tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    correct_pred1 = tf.equal(tf.argmax(model.tc1, 1), tf.argmax(yt1, 1))
    correct1=tf.reduce_sum(tf.cast(correct_pred1,tf.float32))
    accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
    correct_pred2 = tf.equal(tf.argmax(model.tc2, 1), tf.argmax(yt2, 1))
    correct2=tf.reduce_sum(tf.cast(correct_pred2,tf.float32))
    accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))



    graph_op = model.graph_optimize(lr,l2_coef,adlamb)
    update_op = model.optimize(decay_learning_rate,train_layers,adlamb)
    D_op=model.adoptimize(decay_learning_rate,train_layers)
    optimizer=tf.group(graph_op,update_op,D_op)

    
    train_writer=tf.summary.FileWriter('./log/tensorboard')
    train_writer.add_graph(tf.get_default_graph())
    config=projector.ProjectorConfig()
    embedding=config.embeddings.add()
    embedding.tensor_name=model.feature.name
    embedding.metadata_path='domain.csv'
    projector.visualize_embeddings(train_writer,config)
    tf.summary.scalar('C_loss',model.loss)
    tf.summary.scalar('Training Accuracy',source_accuracy)
    tf.summary.scalar('Testing Accuracy',accuracy)
    tf.summary.scalar('Testing Accuracy1',accuracy1)
    tf.summary.scalar('Testing Accuracy2', accuracy2)
    merged=tf.summary.merge_all()




    print ('============================GLOBAL TRAINABLE VARIABLES ============================')
    print (tf.trainable_variables())


    with tf.Session() as sess:
        #
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()



        m = np.zeros([300, 3])
        n = np.zeros([300, 2])
        a = 0
        print("{} Start training...".format(datetime.datetime.now()))

        gd=0
        step = 0
        for epoch in range(30000):

            gd+=1
            lamb=adaptation_factor(gd*1.0/MAX_STEP)
            power=gd/10000
            rate=FLAGS.learning_rate
            tt=pow(0.1,power)
            batch_xs, batch_ys = TRAIN.next_batch(FLAGS.batch_size)
            Tbatch_xs, Tbatch_ys = VALID.next_batch(FLAGS.batch_size)
            Tbatch1_xs, Tbatch1_ys = VALID1.next_batch(FLAGS.batch_size)
            Tbatch2_xs, Tbatch2_ys = VALID2.next_batch(FLAGS.batch_size)
            Ybash_mask = np.ones([FLAGS.batch_size,1])
            Ytbash_mask = np.zeros([FLAGS.batch_size, 1])

            summary,_,closs,dloss,smloss,graphloss=sess.run([merged,optimizer,model.loss,model.JSD,model.smloss,model.loss_graph],
                                                            feed_dict={x_s: batch_xs,x_t: Tbatch_xs,x_t1: Tbatch1_xs,x_t2:Tbatch2_xs,time:[1.0*gd],decay_learning_rate:rate,
                                                                        adlamb:lamb,is_training:True,y: batch_ys,dropout_keep_prob:0.5,yt:Tbatch_ys,yt1:Tbatch1_ys,yt2:Tbatch2_ys,
                                                                        y_mask:Ybash_mask,yt_mask:Ytbash_mask})



	
            step += 1
            if gd%100==0:
                epoch=gd/(72357/100)
                print ('lambda: ',lamb)
                print ('rate: ',rate)
                print ('Epoch {4:<10} Step {2:<10} C_loss {0:<10} D_loss {1:<10} sm_loss {3:<10} Graph_loss {5:<10}'.format(closs,dloss,gd,smloss,epoch,graphloss))
                print("{} Start validation".format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0
                print ('test_iter ',len(TEST.labels))
                for _ in range(int((len(TEST.labels))/5000)):
                    batch_tx, batch_ty = TEST.next_batch(5000)
                    #print TEST.pointer,'   ',TEST.shuffle
                    acc = sess.run(correct, feed_dict={x_t: batch_tx, yt: batch_ty, is_training:True,dropout_keep_prob: 1.})
                    test_acc += acc
                    test_count += 5000
                print (test_acc,test_count)
                test_acc /= test_count
                m[a,0] = test_acc
                test_acc1 = 0.
                test_count1 = 0
                print('test_iter ', len(TEST1.labels))
                for _ in range(int((len(TEST1.labels)) / 2007)):
                    batch_tx, batch_ty = TEST1.next_batch(2007)
                    # print TEST.pointer,'   ',TEST.shuffle
                    acc = sess.run(correct1, feed_dict={x_t1: batch_tx, yt1: batch_ty, is_training: True, dropout_keep_prob: 1.})
                    test_acc1 += acc
                    test_count1 += 2007
                print(test_acc1, test_count1)
                test_acc1 /= test_count1
                m[a, 1] = test_acc1
            #
                test_acc2 = 0.
                test_count2 = 0
                print ('test_iter ',len(TEST2.labels))
                for _ in range(int((len(TEST2.labels))/5000)):
                    batch_tx, batch_ty = TEST2.next_batch(5000)
                    #print TEST.pointer,'   ',TEST.shuffle
                    acc = sess.run(correct2, feed_dict={x_t2: batch_tx, yt2: batch_ty, is_training:True,dropout_keep_prob: 1.})
                    test_acc2 += acc
                    test_count2 += 5000
                print (test_acc2,test_count2)
                test_acc2 /= test_count2
                m[a, 2] = test_acc2
                n[a, 0] = closs
                a= a+1



                print("{} Validation Accuracy = {:.4f} Accuracy = {:.4f} Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc,test_acc1,test_acc2))

                if gd%10000==0 and gd>0:
                    pass
                #print("{} Saving checkpoint of model...".format(datetime.datetime.now()))





if __name__ == '__main__':
    tf.app.run()
