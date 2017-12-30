from __future__ import print_function
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
rng = np.random

def inputdata(filepath):
    filelist = glob.glob(filepath)
    
    inputdata = []
    label = []
    for afile in filelist:
        f = open(afile, "r")
        for s_line in iter(f):
            ll = s_line.split(',')
            iv = []
            for i in range(len(ll)):
                if i == 0:
                    label.append([float(ll[i])])
                else:
                    iv.append(float(ll[i]))
            inputdata.append(iv)
    return np.asarray(inputdata),np.asarray(label)




# Parameters
learning_rate = 0.01
training_epochs = 100
display_step = 50

# Training Data
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                       7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])


bigX,bigY = inputdata("./*.csv")
bigY = normalize(bigY,norm = 'max',axis = 0)
bigX = normalize(bigX,norm = 'max',axis = 0)


n_samples = bigX.shape[0]
# tf Graph Input
X = tf.placeholder("float",[None,50])
Y_ = tf.placeholder("float",[None,6])



# Set model weights
W = tf.Variable(tf.zeros([50,6]), name="weight")
b = tf.Variable(tf.zeros([6]), name="bias")

pred = tf.nn.softmax(matmul(X, W) + b)


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







#The rest is for linear model.
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    # Fit all training data
    for epoch in range(training_epochs):
        #for (x, y) in zip(bigX, bigY):
            #print("this is updated W")
            #print(sess.run(W))
        sess.run(optimizer, feed_dict={X: bigX, Y_: bigY})

        # Display logs per epoch step
        #c = 0
        if (epoch+1) % display_step == 0:
            #for(tx,ty) in zip(bigX,bigY):
            c = sess.run(cost, feed_dict={X: bigX, Y_:bigY})   
            print("Epoch:", '%04d' % (epoch+1), "cost=", c, \
                    "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    
    #exit()
    #training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    #print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    #plt.plot(train_X, train_Y, 'ro', label='Original data')
    #plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    #plt.legend()
    #plt.show()

    # Testing example, as requested (Issue #2)
    #test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    #test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    test_X = bigX
    test_Y = bigY

    print("Testing... (Mean square loss Comparison)")
    #testing_cost = 0
    p = sess.run(pred, feed_dict={X: bigX})
    print("prediction",p)
    print("W",sess.run(W))
    testing_cost = sess.run(
            tf.pow(pred-Y_, 2),
            feed_dict={X: bigX, Y_: bigY})  # same function as cost above
    print("Testing cost=", testing_cost)
    #print("Absolute mean square loss difference:", abs(
    #    training_cost - testing_cost))

    #plt.plot(test_X, test_Y, 'bo', label='Testing data')
    #plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    #plt.legend()
    #plt.show()

