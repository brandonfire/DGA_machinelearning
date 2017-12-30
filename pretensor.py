from __future__ import print_function
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import sys

rng = np.random

if len(sys.argv) != 2:
    print("Error: please use $",sys.argv[0],"{cross validation fold}")

fold = int(sys.argv[1])

def inputdata(filen,Total_f,fth):   
    inputdata = []
    f = open(filen, "r")
    for s_line in iter(f):
            ll = s_line.split(',')
            iv = []
            for i in range(len(ll)):
                if i == 0:
                    pass
                else:
                    iv.append(float(ll[i]))
            inputdata.append(iv)
    f.close()
    itemnumber = 40/Total_f
    start = itemnumber*fth
    testdata = inputdata[start:start+itemnumber]
    
    traindata = inputdata[0:start]+inputdata[(start+itemnumber):]
    
    return np.asarray(traindata),np.asarray(testdata)

def inputlabel(labelfile,Total_f,fth):
    f = open(labelfile, "r")
    label = []
    for s_line in iter(f):
        ll = s_line.split(',')
        ls = []
        for l in ll:
            ls.append(float(l))
        label.append(ls)
    itemnumber = 40/Total_f
    start = itemnumber*fth
    testdata = label[start:start+itemnumber]
    
    traindata = label[0:start]+label[(start+itemnumber):]  
    return np.asarray(traindata),np.asarray(testdata)

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Training Data
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                       7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

for fo in range(fold):
    bigX,tx = inputdata("dataAnalysis_16.csv",fold,fo)
    bigY,ty = inputlabel("JHUoutput3.csv",fold,fo)
    bigX = normalize(bigX,norm = 'max',axis = 0)


    n_samples = bigX.shape[0]
    # tf Graph Input
    X = tf.placeholder("float",[None,16])
    Y_ = tf.placeholder("float",[None,3])



    # Set model weights
    W = tf.Variable(tf.zeros([16,3]), name="weight")
    b = tf.Variable(tf.zeros([3]), name="bias")

    pred = tf.nn.softmax(tf.matmul(X, W) + b)


    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_*tf.log(pred), reduction_indices=[1]))
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Initialize the variables (i.e. assign their default value)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()





    init = tf.global_variables_initializer()

    for epoch in range(training_epochs):
        sess.run(train_step, feed_dict = {X: bigX, Y_: bigY})



    correctcount = tf.equal(tf.argmax(pred,1), tf.argmax(Y_,1))

    accuracy = tf.reduce_mean(tf.cast(correctcount, tf.float32))

    #print("model")
    #print(sess.run(W))
    #print(sess.run(b))
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "./model.ckpt")
    #print("model saved!")
    print("accuracy is:")

    print(sess.run(accuracy, feed_dict = {X: tx, Y_: ty} ))




