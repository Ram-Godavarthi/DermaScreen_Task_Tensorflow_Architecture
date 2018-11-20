import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
import network_architectures

#read the dataset
def data_Set():
    df_train = pd.read_csv('dataset/train.csv')
    df_train = pd.get_dummies(df_train,columns=["label"])
    df_features = df_train.iloc[:, :-10].values
    df_label = df_train.iloc[:, -10:].values
    print(df_features.shape)


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_label,
                                                    test_size = 0.2,
                                                    random_state = 1212)

    X_test,X_validation,y_test,y_validation = train_test_split(X_test,
                                                               y_test,
                                                               test_size=0.5,
                                                               random_state=0)
    print(df_label.shape)


    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale

    def reformat(dataset, labels):
      dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
      labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
      return dataset, labels

    train_dataset, train_labels = reformat(X_train, y_train)
    valid_dataset, valid_labels = reformat(X_validation, y_validation)
    test_dataset , test_labels  = reformat(X_test, y_test)


    print ('Training set   :', train_dataset.shape, train_labels.shape)
    print ('Validation set :', valid_dataset.shape, valid_labels.shape)
    print ('Test set       :', test_dataset.shape, test_labels.shape)



    X_train      = np.pad(train_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(valid_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(test_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    print ('Training set after padding 2x2    :', X_train.shape, train_labels.shape)
    print ('Validation set after padding 2x2  :', X_validation.shape, valid_labels.shape)
    print ('Test set after padding 2x2        :', X_test.shape, test_labels.shape)




    x = tf.placeholder(tf.float32, shape=[None,32,32,1])
    y_ = tf.placeholder(tf.int32, (None))




## Note : Here you can import any network that is defined in "network_architectures.py" file

    # import the network
    #logits = network_architectures.LeNet_5(x)
    logits = network_architectures.MobileNet(x)

    #Softmax with cost function implementation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss_operation)


    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


    EPOCHS = 2
    BATCH_SIZE = 128

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training started with dataset - ", num_examples)
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})

            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver = tf.train.Saver()
        save_path = saver.save(sess, 'C:/Users/godavart/Desktop/Project_dermascreen/tmp/model.ckpt')
        print("Model saved %s " % save_path)

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


if __name__ =='__main__':
    train_setup =data_Set()
