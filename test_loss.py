import tensorflow as tf
import pascal_voc
import numpy as np

import loss_v6 as loss_fn

predict_array = tf.random_uniform([2,7*7*30],dtype=tf.float64)
labels_array = tf.random_uniform([2,7,7,25],dtype=tf.float64)
batch_size = -1
cell_size = 7
boxes_per_cell = 2
num_class = 20


# Load training and eval data
pascal = pascal_voc.pascal_voc('train')
images, labels = pascal.get()
images = np.float64(images)
labels = np.float64(labels)

print(images)
# normalize_labels(labels)
train_data = images  # Returns np.array
train_labels = labels


# loss = loss_fn.loss_fn(predict_array, labels_array)
# sess = tf.Session()
# print(sess.run(loss))