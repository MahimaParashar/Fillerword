from utils import parse_data
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle as pkl
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import keras

# import pdb; pdb.set_trace()

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def train(features, labels):
    X_train, X_test, y_tr, y_tst = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    y_train = one_hot_encode(y_tr.astype(int))
    y_test = one_hot_encode(y_tst.astype(int))

    training_epocs = 10
    n_dim = X_train.shape[1]
    n_classes = 2
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_two], mean=0, stddev=sd))
    b_1 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

    # W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
    # b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
    # h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

    W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
    b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
    pred_y = tf.nn.softmax(tf.matmul(h_1, W) + b)
    init = tf.global_variables_initializer()

    cost_function = -tf.reduce_sum(y_train * tf.log(pred_y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost_history = np.empty(shape=[1], dtype=float)
    y_true, y_pred = None, None
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epocs):
            _, cost = sess.run([optimizer, cost_function], feed_dict={X: X_train, Y: y_train})
            cost_history = np.append(cost_history, cost)

            y_pred = sess.run(tf.argmax(pred_y, 1), feed_dict={X: X_train})
            y_true = sess.run(tf.argmax(y_train, 1))
            print(epoch+1,": Test accuracy: ", round(sess.run(accuracy, feed_dict={X: X_test, Y: y_test}), 3))




def keras_train(data,labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25, random_state = 42)

    epochs = 10
    n_dim = X_train.shape[1]
    n_hidden_units_one = 280
    n_hidden_units_two = 300
    sd = 1 / np.sqrt(n_dim)
    learning_rate = 0.001

    model = Sequential([
        Dense(n_hidden_units_one, input_shape = (n_dim,), kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=sd)),
        Activation('tanh'),
        Dense(n_hidden_units_two, kernel_initializer = keras.initializers.RandomNormal(mean=0, stddev=sd)),
        Activation('sigmoid'),
        Dense(1),
        Activation('sigmoid')
    ])
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train,y_train,epochs = epochs, batch_size = 32)

    score = model.evaluate(X_test,y_test, batch_size = 32)
    print (score)
    

def main():
    # data,labels = parse_data("./TrainingDataUmm")

    with open("./data/umm_balanced.pkl","rb") as pk:
        features, labels = pkl.load(pk)
        keras_train(features, labels)



if __name__ == '__main__':
    main()