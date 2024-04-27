#import library
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches, convert_to_one_hot
from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split


import tensorflow.compat.v1 as tf




def create_placeholders(n_x1, n_x2, n_y):
    tf.disable_eager_execution()
    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
    isTraining = tf.placeholder_with_default(True, shape=(), name='isTraining')
    x_pure = tf.placeholder(tf.float32, shape=[None, n_x1], name="x_pure")
    x_mixed = tf.placeholder(tf.float32, shape=[None, n_x2], name="x_mixed")
    y = tf.placeholder(tf.float32, shape=[None, n_y], name="Y")
    return x_pure, x_mixed, y, isTraining, keep_prob


def initialize_parameters():
    tf.compat.v1.set_random_seed(1)
    initializer = tf.keras.initializers.VarianceScaling(seed=1)

    x_w1 = tf.compat.v1.get_variable("x_w1", [224, 256], initializer=initializer)
    x_b1 = tf.compat.v1.get_variable("x_b1", [256], initializer=tf.zeros_initializer())

    x_w2 = tf.compat.v1.get_variable("x_w2", [256, 128], initializer=initializer)
    x_b2 = tf.compat.v1.get_variable("x_b2", [128], initializer=tf.zeros_initializer())

    x_w3 = tf.compat.v1.get_variable("x_w3", [128, 32], initializer=initializer)
    x_b3 = tf.compat.v1.get_variable("x_b3", [32], initializer=tf.zeros_initializer())

    x_w4 = tf.compat.v1.get_variable("x_w4", [32, 12], initializer=initializer)
    x_b4 = tf.compat.v1.get_variable("x_b4", [12], initializer=tf.zeros_initializer())

    x_dew1 = tf.compat.v1.get_variable("x_dew1", [12, 32], initializer=initializer)
    x_deb1 = tf.compat.v1.get_variable("x_deb1", [32], initializer=tf.zeros_initializer())

    x_dew2 = tf.compat.v1.get_variable("x_dew2", [32, 128], initializer=initializer)
    x_deb2 = tf.compat.v1.get_variable("x_deb2", [128], initializer=tf.zeros_initializer())

    x_dew3 = tf.compat.v1.get_variable("x_dew3", [128, 256], initializer=initializer)
    x_deb3 = tf.compat.v1.get_variable("x_deb3", [256], initializer=tf.zeros_initializer())

    x_dew4 = tf.compat.v1.get_variable("x_dew4", [256, 224], initializer=initializer)
    x_deb4 = tf.compat.v1.get_variable("x_deb4", [224], initializer=tf.zeros_initializer())

    parameters = {
        "x_w1": x_w1,
        "x_b1": x_b1,
        "x_w2": x_w2,
        "x_b2": x_b2,
        "x_w3": x_w3,
        "x_b3": x_b3,
        "x_w4": x_w4,
        "x_b4": x_b4,
        "x_dew1": x_dew1,
        "x_deb1": x_deb1,
        "x_dew2": x_dew2,
        "x_deb2": x_deb2,
        "x_dew3": x_dew3,
        "x_deb3": x_deb3,
        "x_dew4": x_dew4,
        "x_deb4": x_deb4
    }

    return parameters



    
def mynetwork(x_pure, x_mixed, parameters, isTraining, keep_prob, momentum = 0.9):
    
    with tf.name_scope("x_layer_1"):
        
         x_pure_z1 = tf.matmul(x_pure, parameters['x_w1']) + parameters['x_b1'] 
         x_pure_z1_bn = tf.layers.batch_normalization(x_pure_z1, axis=-1, momentum=momentum, training=isTraining, name='l1', reuse=tf.AUTO_REUSE)

         x_pure_z1_do = tf.nn.dropout(x_pure_z1_bn, keep_prob)  
         x_pure_a1 = tf.nn.tanh(x_pure_z1_do)

         x_mixed_z1 = tf.matmul(x_mixed, parameters['x_w1']) + parameters['x_b1']                       
         x_mixed_z1_bn = tf.layers.batch_normalization(x_mixed_z1, axis =- 1, momentum = momentum, training = isTraining, name = 'l1', reuse = True)
         x_mixed_z1_do = tf.nn.dropout(x_mixed_z1_bn, keep_prob)          
         x_mixed_a1 = tf.nn.tanh(x_mixed_z1_do)
         
    with tf.name_scope("x_layer_2"):
        
         x_pure_z2 = tf.matmul(x_pure_a1, parameters['x_w2']) + parameters['x_b2']                                         
         x_pure_z2_bn = tf.layers.batch_normalization(x_pure_z2, axis =-1,  momentum = momentum, training = isTraining, name = 'l2',reuse=tf.AUTO_REUSE)
         x_pure_a2 = tf.nn.tanh(x_pure_z2_bn)
         
         x_mixed_z2 = tf.matmul(x_mixed_a1, parameters['x_w2']) + parameters['x_b2']                             
         x_mixed_z2_bn = tf.layers.batch_normalization(x_mixed_z2, axis = -1, momentum = momentum, training = isTraining, name = 'l2', reuse = True)
         x_mixed_a2 = tf.nn.tanh(x_mixed_z2_bn)
    
    with tf.name_scope("x_layer_3"):
        
         x_pure_z3 = tf.matmul(x_pure_a2, parameters['x_w3']) + parameters['x_b3']             
         x_pure_z3_bn = tf.layers.batch_normalization(x_pure_z3, axis = -1,  momentum = momentum, training = isTraining, name = 'l3',reuse=tf.AUTO_REUSE)
         x_pure_a3 = tf.nn.relu(x_pure_z3_bn)
         
         x_mixed_z3 = tf.matmul(x_mixed_a2, parameters['x_w3']) + parameters['x_b3']                                
         x_mixed_z3_bn = tf.layers.batch_normalization(x_mixed_z3, axis = -1, momentum = momentum, training = isTraining, name = 'l3', reuse = True)
         x_mixed_a3 = tf.nn.relu(x_mixed_z3_bn)

    with tf.name_scope("x_layer_4"):
    
         x_pure_z4 = tf.add(tf.matmul(x_pure_a3, parameters['x_w4']), parameters['x_b4'])
         abundances_pure = tf.nn.softmax(x_pure_z4)
         
         x_mixed_z4 = tf.add(tf.matmul(x_mixed_a3, parameters['x_w4']), parameters['x_b4'])   
         abundances_mixed = tf.nn.softmax(x_mixed_z4)
         
    with tf.name_scope("x_de_layer_1"):
        
         x_mixed_de_z1 = tf.matmul(abundances_mixed, parameters['x_dew1']) + parameters['x_deb1']                                          
         x_mixed_de_z1_bn = tf.layers.batch_normalization(x_mixed_de_z1, axis = -1,  momentum = momentum, training = isTraining)
         x_mixed_de_a1 = tf.nn.sigmoid(x_mixed_de_z1_bn)

    with tf.name_scope("x_de_layer_2"):
        
         x_mixed_de_z2 = tf.matmul(x_mixed_de_a1, parameters['x_dew2']) + parameters['x_deb2']                                           
         x_mixed_de_z2_bn = tf.layers.batch_normalization(x_mixed_de_z2, axis = -1,  momentum = momentum, training = isTraining)
         x_mixed_de_a2 = tf.nn.sigmoid(x_mixed_de_z2_bn)

    with tf.name_scope("x_de_layer_3"):
        
         x_mixed_de_z3 = tf.matmul(x_mixed_de_a2, parameters['x_dew3']) + parameters['x_deb3']                                          
         x_mixed_de_z3_bn = tf.layers.batch_normalization(x_mixed_de_z3, axis = -1,  momentum = momentum, training = isTraining)
         x_mixed_de_a3 = tf.nn.sigmoid(x_mixed_de_z3_bn)

    with tf.name_scope("x_de_layer_4"):
        
         x_mixed_de_z4 = tf.matmul(x_mixed_de_a3, parameters['x_dew4']) + parameters['x_deb4']                                         
         x_mixed_de_z4_bn = tf.layers.batch_normalization(x_mixed_de_z4, axis = -1,  momentum = momentum, training = isTraining)
         x_mixed_de_a4 = tf.nn.sigmoid(x_mixed_de_z4_bn)
         
    l2_loss =  tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2']) + tf.nn.l2_loss(parameters['x_w3']) + tf.nn.l2_loss(parameters['x_w4'])\
               + tf.nn.l2_loss(parameters['x_dew1']) + tf.nn.l2_loss(parameters['x_dew2']) + tf.nn.l2_loss(parameters['x_dew3']) + tf.nn.l2_loss(parameters['x_dew4'])

   
    return x_pure_z4, abundances_mixed, x_mixed_de_a4, l2_loss, abundances_pure, abundances_mixed

def mynetwork_optimaization(y_est, y_re, r1, r2, l2_loss, reg, learning_rate, global_step):
    
    with tf.name_scope("cost"):
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_est, labels = y_re))\
                + reg * l2_loss + 1 * tf.reduce_mean(tf.pow(r1 - r2, 2))
                
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,  global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def train_mynetwork(x_pure_set, x_mixed_set, y_train, y_test, learning_rate_base = 0.001, beta_reg = 0.002, num_epochs = 200, minibatch_size = 8000, print_cost = True):
    
    ops.reset_default_graph()                         
    tf.compat.v1.set_random_seed(1)

                           
    seed = 1                                    
    (m,n_x1)= x_pure_set.shape         
    (m1,n_x2)=x_mixed_set.shape
    (m, n_y) = y_train.shape                            
    
    costs = []                                        
    costs_dev = []
    train_acc = []
    val_acc = []
    
    x_train_pure, x_train_mixed, y, isTraining, keep_prob = create_placeholders(n_x1, n_x2, n_y) 

    parameters = initialize_parameters()
    
    with tf.name_scope("network"):
         x_pure_layer, x_mixed_layer, x_mixed_de_layer, l2_loss, abundances_pure, abundances_mixed = mynetwork(x_train_pure, x_train_mixed, parameters, isTraining, keep_prob)
         
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
    learning_rate_base, global_step, m/minibatch_size, 0.99)
    
    with tf.name_scope("optimization"):
         cost, optimizer = mynetwork_optimaization(x_pure_layer, y, x_mixed_de_layer, x_train_mixed, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         accuracy = tf.losses.absolute_difference(labels = y, predictions = abundances_pure)
         
    init = tf.global_variables_initializer()

   
    with tf.Session() as sess:
        
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m1 / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x_pure_set, x_mixed_set, y_train, minibatch_size, seed)
            i=0
            for minibatch in minibatches:
                # Select a minibatch
                (batch_x1, batch_x2, batch_y) = minibatch
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x_train_pure: batch_x1, x_train_mixed: batch_x2, y: batch_y, isTraining: True, keep_prob: 0.9})
                epoch_cost += minibatch_cost
                epoch_acc += minibatch_acc
                i=i+1
               
            epoch_cost_f = epoch_cost / (num_minibatches + 1)    
            epoch_acc_f = epoch_acc / (num_minibatches + 1)

            

            abund, re, epoch_cost_dev, epoch_acc_dev = sess.run([abundances_pure, x_mixed_de_layer, cost, accuracy], feed_dict={x_train_pure: x_mixed_set, x_train_mixed: x_mixed_set, y: y_test, isTraining: True, keep_prob: 1})

            if print_cost == True:
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost_f, epoch_cost_dev, epoch_acc_f, epoch_acc_dev))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost_f)
                train_acc.append(epoch_acc_f)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)

        # plot the cost      
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        return parameters , val_acc, abund



# Load the data
Pure_TrSet = scio.loadmat('TNNLS_Data/cuprite/pure2D.mat')
Mixed_TrSet = scio.loadmat('TNNLS_Data/cuprite/mixed2D.mat')
TrLabel = scio.loadmat('TNNLS_Data/cuprite/mixed_labels.mat')
TeLabel = scio.loadmat('TNNLS_Data/cuprite/pure_labels.mat')

# Extract the data from the loaded mat files
Pure_TrSet = Pure_TrSet['cube_2d']
Mixed_TrSet = Mixed_TrSet['cube_2d']
TrLabel = TrLabel['cube_2d']
TeLabel = TeLabel['cube_2d']
print("pure_Trset size is ",Pure_TrSet.shape);
# Transpose the label matrices
Y_train = TrLabel
Y_test = TeLabel

# Function to split 2D data while preserving pairing of data points
def split_data(X, y, test_size=0.2, random_state=42):
    # Generate indices for splitting
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Split the data using the generated indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# Split the mixed2D data and labels while preserving pairing
Mixed_TrSet_train,Mixed_TrSet_test,Yn_train,Yn_test = split_data(Mixed_TrSet, TrLabel)

# Train the network with the train set
parameters, val_acc, abund_train = train_mynetwork(Pure_TrSet, Mixed_TrSet, Y_train, Y_test)


# Evaluate the model on the test set
with tf.Session() as sess:
    x_train_pure, x_train_mixed, y, isTraining, keep_prob = create_placeholders(Pure_TrSet.shape[1], Mixed_TrSet_test.shape[1], Y_test.shape[1])
    x_pure_layer, _, _, _, _, abund_test = mynetwork(x_train_pure, x_train_mixed, parameters, isTraining, keep_prob)
    accuracy_test = tf.losses.absolute_difference(labels=y, predictions=abund_test)

    # Load the trained parameters
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())


    # Evaluate accuracy on the test set
    accuracy_value = sess.run(accuracy_test, feed_dict={x_train_pure: Pure_TrSet, x_train_mixed: Mixed_TrSet_test, y: Yn_test, isTraining: False, keep_prob: 1.0})
    print("Test Accuracy:", accuracy_value*100)

