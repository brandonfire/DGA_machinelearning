from __future__ import print_function
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
import time
rng = np.random
tf.logging.set_verbosity(tf.logging.INFO)


alphacheck = {
'A':1,
'B':2,
'C':3,
'D':4,
'E':5,
'F':6,
'G':7,
'H':8,
'I':9,
'J':10,
'K':11,
'L':12,
'M':13,
'N':14,
'O':15,
'P':16,
'Q':17,
'R':18,
'S':19,
'T':20,
'U':21,
'V':22,
'W':23,
'X':24,
'Y':25,
'Z':26,
'\r':27,
'0':100,
'1':101,
'2':102,
'3':103,
'4':104,
'5':105,
'6':106,
'7':107,
'8':108,
'9':109,
'-':200}



def inputdata(filename):   
    trainingdata = []
    #for afile in filelist:
    f = open(filename, "r")
    for s_line in iter(f):
            s_line = s_line.upper()
            sz_dga_matrix = []
            for c in s_line.rstrip('\n'):
                sz_dga_matrix.append(alphacheck[c])
            if len(sz_dga_matrix) < 16:
                for _ in range(16-len(sz_dga_matrix)):
                    sz_dga_matrix.append(0)
            else:
                sz_dga_matrix = sz_dga_matrix[:16]
            trainingdata.append(np.asarray(sz_dga_matrix))
    return np.asarray(trainingdata)

def inputlabel(labelfile):
    f = open(labelfile, "r")
    label = []
    for s_line in iter(f):
        label.append(s_line)
    return np.asarray(label)

def cnn_model_fn(features, labels, mode):

    input_layer = tf.reshape(features["x"], [-1, 4, 4, 1])
    #Input Tensor Shape: [batch_size, 4, 3, 1]
    # Output Tensor Shape: [batch_size, 4, 3, 16]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[3, 3],
      padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #compute 32 features
    #conv2 = tf.layers.conv2d(
    #  inputs=conv1,
    #  filters=32,
    #  kernel_size=[3, 3],
    #  padding="same",
    #activation=tf.nn.relu)

    
    conv3flat = tf.reshape(pool1, [-1, 2 * 2 * 16])
    dense = tf.layers.dense(inputs=conv3flat, units=20, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout, units=20, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2, units=26)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, model_dir="./tmp/L2cryptomax2")
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=26)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    #learning_rate = 0.01
    training_epochs = 2000
    display_step = 50
    tdata = inputdata("LargeCryto.txt")
    tlabel = inputlabel("DGAlabel1st.csv")
    bigX = np.asarray(tdata[:-1],dtype=np.float32)
    bigY = np.asarray(tlabel[1:],dtype=np.int32)
    n_samples = bigX.shape[0]
    DGA_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="./tmp/L2cryptomax2")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5000)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": bigX},
      y=bigY,
      batch_size=100,
      num_epochs=None, shuffle=True)
    DGA_classifier.train(
      input_fn=train_input_fn,
      steps=20000,hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": bigX},
      y=bigY,
      num_epochs=1,
      shuffle=False)
    eval_results = DGA_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)






