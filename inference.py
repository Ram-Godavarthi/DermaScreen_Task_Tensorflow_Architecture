import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import demo_script
import network_architectures
from demo_script import data_Set
from config import generate_config,default


def restore_func():
    df_test = pd.read_csv('dataset/test.csv')
    df_test.head()

    df_test = df_test.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    print('Submission Test set       :', df_test.shape)

    submission_test = np.pad(df_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    print('Submission data after padding 2x2 :', submission_test.shape)

    x = tf.placeholder(tf.float32, shape=[None,32,32,1])
    y_ = tf.placeholder(tf.int32, (None))
    #logits = cfg.network

    selected_network = cfg.network
    if selected_network == "LeNet_5":
        logits = network_architectures.LeNet_5(x)
    else:
        logits = network_architectures.MobileNet_4(x)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, 'C:/Users/godavart/Desktop/Project_dermascreen/tmp/model.ckpt')
        print("Model restored.")
        Z = logits.eval(feed_dict={x: submission_test})
        y_pred = np.argmax(Z, axis=1)
        print(len(y_pred))

        # Write into a CSV file with columns ImageId & Label
        submission = pd.DataFrame({
            "ImageId": list(range(1, len(y_pred) + 1)),
            "Label": y_pred
        })
    submission.to_csv('ram.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Access the network')
    parser.add_argument('--network', type=str, default= default.network, help='network name')
    parser.add_argument('--lr', type=float, default= default.lr , help = 'learning rate')
    parser.add_argument('--image_size', type=int, default=default.image_size, help='input image size')
    return parser.parse_args()


if __name__ =='__main__':
    args = parse_args()
    cfg = generate_config(vars(args))
    restore_setup = restore_func()

    image_size = cfg.image_size
    num_labels = 10
    num_channels = 1