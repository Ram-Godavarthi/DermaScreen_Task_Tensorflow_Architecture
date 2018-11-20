import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet_5(x):
    # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional.
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc1 = flatten(pool_2)

    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)

    # outputs 10 classes
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=0.1))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

def MobileNet(x):
    conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1,8], mean=0, stddev=0.1))
    conv1_b = tf.Variable(tf.zeros(8))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 2, 2, 1], padding='VALID', name='conv1') + conv1_b
    relu1 = tf.nn.relu(conv1)
    norm1 = tf.layers.batch_normalization(relu1, name='norm1')
    #pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu) -> Lrn -> Pool
    conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 8, 8], mean=0, stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(8))
    conv2 = tf.nn.conv2d(norm1, conv2_w, strides=[1,1,1,1], padding='VALID', name='conv2') + conv2_b
    relu2 = tf.nn.relu(conv2)
    norm2 = tf.layers.batch_normalization(relu2, name='norm2')
    #pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 8, 16], mean=0, stddev=0.1))
    conv3_b = tf.Variable(tf.zeros(16))
    conv3 = tf.nn.conv2d(norm2, conv3_w,  strides=[1,1,1,1], padding='VALID', name='conv3') + conv3_b
    relu3 = tf.nn.relu(conv3)
    norm3 = tf.layers.batch_normalization(relu3, name='norm3')

    # 4th Layer: Conv (w ReLu)
    conv4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 16], mean=0, stddev=0.1))
    conv4_b = tf.Variable(tf.zeros(16))
    conv4 = tf.nn.conv2d(norm3, conv4_w,  strides=[1, 2, 2, 1],padding='VALID', name='conv4') + conv4_b
    relu4 = tf.nn.relu(conv4)
    norm4 = tf.layers.batch_normalization(relu4, name='norm4')

    # 6th Layer: Flatten -> FC
    fc1_w = tf.Variable(tf.truncated_normal(shape=(576, 10), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(10))
    flatten_output = flatten(norm4)
    logits = tf.matmul(flatten_output, fc1_w) + fc1_b
    return logits

if __name__ =='__main__':
    net = LeNet_5(x)
    net1 = MobileNet(x) # taken only top 4 layers
    print(net)
    print(net1)

