import tensorflow as tf
import numpy as np
import config as cfg
import old_loss as loss_fn
print("TensorFlow Version is: ")
print(tf.__version__)
import functools
import pascal_voc

#initializing variables
cell_size = 7
boxes_per_cell = 2
initial_learning_rate = cfg.LEARNING_RATE
decay_steps = cfg.DECAY_STEPS
decay_rate = cfg.DECAY_RATE
staircase = cfg.STAIRCASE

config = tf.ConfigProto()
config.gpu_options.allocator_type ='BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 448, 448, 3])
    input_layer = tf.cast(input_layer, tf.float64)

    input_layer = tf.pad(input_layer, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        strides=(2, 2),
        kernel_size=[7, 7],
        kernel_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.zeros_initializer(),
        padding="valid",
        activation=None, name='conv1')  # CHANGED ACTIVATION FROM RELU TO LEAKY RELU

    conv1_activated = tf.nn.leaky_relu(conv1, alpha=0.1)

    # with tf.variable_scope("conv1", reuse=True):
    #     weights_conv1 = tf.get_variable("kernel")
    #     biases_conv1=tf.get_variable("bias")

    # Pooling Layer #1
    pool1 = tf.nn.max_pool(conv1_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv2')

    conv2_activated = tf.nn.leaky_relu(conv2, alpha=0.1)

    #     with tf.variable_scope("conv2", reuse=True):
    #         weights_conv2 = tf.get_variable("kernel")
    #         biases_conv2=tf.get_variable("bias")

    # Pooling Layer #2
    #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same', name ='pool2')
    pool2 = tf.nn.max_pool(conv2_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv3')

    conv3_activated = tf.nn.leaky_relu(conv3, alpha=0.1)
    #     with tf.variable_scope("conv3", reuse=True):
    #         weights_conv3 = tf.get_variable("kernel")
    #         biases_conv3=tf.get_variable("bias")

    conv4 = tf.layers.conv2d(
        inputs=conv3_activated,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv4')

    conv4_activated = tf.nn.leaky_relu(conv4, alpha=0.1)
    #     with tf.variable_scope("conv4", reuse=True):
    #         weights_conv4 = tf.get_variable("kernel")
    #         biases_conv4=tf.get_variable("bias")

    conv5 = tf.layers.conv2d(
        inputs=conv4_activated,
        filters=256,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv5')

    conv5_activated = tf.nn.leaky_relu(conv5, alpha=0.1)
    #     with tf.variable_scope("conv5", reuse=True):
    #         weights_conv5 = tf.get_variable("kernel")
    #         biases_conv5=tf.get_variable("bias")

    conv6 = tf.layers.conv2d(
        inputs=conv5_activated,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv6')

    conv6_activated = tf.nn.leaky_relu(conv6, alpha=0.1)
    #     with tf.variable_scope("conv6", reuse=True):
    #         weights_conv6 = tf.get_variable("kernel")
    #         biases_conv6=tf.get_variable("bias")

    #     pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2, padding='same', name ='pool3')
    pool3 = tf.nn.max_pool(conv6_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    conv7 = tf.layers.conv2d(
        inputs=pool3,
        filters=256,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv7')

    conv7_activated = tf.nn.leaky_relu(conv7, alpha=0.1)

    #     with tf.variable_scope("conv7", reuse=True):
    #         weights_conv7 = tf.get_variable("kernel")
    #         biases_conv7=tf.get_variable("bias")

    conv8 = tf.layers.conv2d(
        inputs=conv7_activated,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv8')

    conv8_activated = tf.nn.leaky_relu(conv8, alpha=0.1)

    #     with tf.variable_scope("conv8", reuse=True):
    #         weights_conv8 = tf.get_variable("kernel")
    #         biases_conv8=tf.get_variable("bias")

    conv9 = tf.layers.conv2d(
        inputs=conv8_activated,
        filters=256,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv9')

    conv9_activated = tf.nn.leaky_relu(conv9, alpha=0.1)

    #     with tf.variable_scope("conv9", reuse=True):
    #         weights_conv9 = tf.get_variable("kernel")
    #         biases_conv9=tf.get_variable("bias")

    conv10 = tf.layers.conv2d(
        inputs=conv9_activated,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv10')

    conv10_activated = tf.nn.leaky_relu(conv10, alpha=0.1)

    #     with tf.variable_scope("conv10", reuse=True):
    #         weights_conv10 = tf.get_variable("kernel")
    #         biases_conv10=tf.get_variable("bias")
    conv11 = tf.layers.conv2d(
        inputs=conv10_activated,
        filters=256,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv11')

    conv11_activated = tf.nn.leaky_relu(conv11, alpha=0.1)

    #     with tf.variable_scope("conv11", reuse=True):
    #         weights_conv11 = tf.get_variable("kernel")
    #         biases_conv11=tf.get_variable("bias")

    conv12 = tf.layers.conv2d(
        inputs=conv11_activated,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv12')

    conv12_activated = tf.nn.leaky_relu(conv12, alpha=0.1)

    #     with tf.variable_scope("conv12", reuse=True):
    #         weights_conv12 = tf.get_variable("kernel")
    #         biases_conv12=tf.get_variable("bias")

    conv13 = tf.layers.conv2d(
        inputs=conv12_activated,
        filters=256,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv13')

    conv13_activated = tf.nn.leaky_relu(conv13, alpha=0.1)

    #     with tf.variable_scope("conv13", reuse=True):
    #         weights_conv13 = tf.get_variable("kernel")
    #         biases_conv13=tf.get_variable("bias")

    conv14 = tf.layers.conv2d(
        inputs=conv13_activated,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv14')

    conv14_activated = tf.nn.leaky_relu(conv14, alpha=0.1)

    #     with tf.variable_scope("conv14", reuse=True):
    #         weights_conv14 = tf.get_variable("kernel")
    #         biases_conv14=tf.get_variable("bias")

    conv15 = tf.layers.conv2d(
        inputs=conv14_activated,
        filters=512,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv15')

    conv15_activated = tf.nn.leaky_relu(conv15, alpha=0.1)

    #     with tf.variable_scope("conv15", reuse=True):
    #         weights_conv15 = tf.get_variable("kernel")
    #         biases_conv15=tf.get_variable("bias")

    conv16 = tf.layers.conv2d(
        inputs=conv15_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv16')

    conv16_activated = tf.nn.leaky_relu(conv16, alpha=0.1)

    #     with tf.variable_scope("conv16", reuse=True):
    #         weights_conv16 = tf.get_variable("kernel")
    #         biases_conv16=tf.get_variable("bias")

    #     pool4 = tf.layers.max_pooling2d(inputs=conv16, pool_size=[2, 2], strides=2, padding='same', name ='pool4')
    pool4 = tf.nn.max_pool(conv16_activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    conv17 = tf.layers.conv2d(
        inputs=pool4,
        filters=512,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv17')

    conv17_activated = tf.nn.leaky_relu(conv17, alpha=0.1)

    #     with tf.variable_scope("conv17", reuse=True):
    #         weights_conv17 = tf.get_variable("kernel")
    #         biases_conv17=tf.get_variable("bias")

    conv18 = tf.layers.conv2d(
        inputs=conv17_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv18')

    conv18_activated = tf.nn.leaky_relu(conv18, alpha=0.1)

    #     with tf.variable_scope("conv18", reuse=True):
    #         weights_conv18 = tf.get_variable("kernel")
    #         biases_conv18=tf.get_variable("bias")

    conv19 = tf.layers.conv2d(
        inputs=conv18_activated,
        filters=512,
        kernel_size=[1, 1],
        padding="same",
        activation=None, name='conv19')

    conv19_activated = tf.nn.leaky_relu(conv19, alpha=0.1)

    #     with tf.variable_scope("conv19", reuse=True):
    #         weights_conv19 = tf.get_variable("kernel")
    #         biases_conv19=tf.get_variable("bias")

    conv20 = tf.layers.conv2d(
        inputs=conv19_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv20')

    conv20_activated = tf.nn.leaky_relu(conv20, alpha=0.1)

    #     with tf.variable_scope("conv20", reuse=True):
    #         weights_conv20 = tf.get_variable("kernel")
    #         biases_conv20=tf.get_variable("bias")

    conv21 = tf.layers.conv2d(
        inputs=conv20_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv21')

    conv21_activated = tf.nn.leaky_relu(conv21, alpha=0.1)

    #     with tf.variable_scope("conv21", reuse=True):
    #         weights_conv21 = tf.get_variable("kernel")
    #         biases_conv21=tf.get_variable("bias")

    conv21_activated = tf.pad(conv21_activated, np.array([[0, 0], [1, 1], [1, 1], [0,
                                                                                   0]]))
    # Padding is done to make sure the shape goes from 14X14X1024 to 7X7X1024 NOT to 6X6X1024.

    conv22 = tf.layers.conv2d(
        inputs=conv21_activated,
        filters=1024,
        kernel_size=[3, 3],
        strides=(2, 2),
        padding="valid",
        activation=None, name='conv22')

    conv22_activated = tf.nn.leaky_relu(conv22, alpha=0.1)

    #     with tf.variable_scope("conv22", reuse=True):
    #         weights_conv22 = tf.get_variable("kernel")
    #         biases_conv22=tf.get_variable("bias")

    conv23 = tf.layers.conv2d(
        inputs=conv22_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv23')

    conv23_activated = tf.nn.leaky_relu(conv23, alpha=0.1)

    #     with tf.variable_scope("conv23", reuse=True):
    #         weights_conv23 = tf.get_variable("kernel")
    #         biases_conv23=tf.get_variable("bias")

    conv24 = tf.layers.conv2d(
        inputs=conv23_activated,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        activation=None, name='conv24')

    conv24_activated = tf.nn.leaky_relu(conv24, alpha=0.1)

    #     with tf.variable_scope("conv24", reuse=True):
    #         weights_conv24 = tf.get_variable("kernel")
    #         biases_conv24=tf.get_variable("bias")

    conv24_activated_flatten_dim = int(functools.reduce(lambda a, b: a * b, conv24_activated.get_shape()[1:]))
    conv24_activated_flat = tf.reshape(tf.transpose(conv24_activated, (0, 3, 1, 2)), [-1, conv24_activated_flatten_dim])

    # Flatten tensor into a batch of vectors

    #     conv24_flat = tf.reshape(conv24, [-1, 7 * 7 * 1024])         ### CHECK THE BATCH SIZE THING? ALSO CHECK PADDING PARAMETER. ALSO CHECK ACTIVATIONS OF LAYERS. ALSO CHECK -1 in reshape.

    # Dense Layer
    dense1 = tf.layers.dense(inputs=conv24_activated_flat, units=512, activation=None, name='dense1')

    dense1_activated = tf.nn.leaky_relu(dense1, alpha=0.1)

    #     with tf.variable_scope("dense1", reuse=True):
    #         weights_dense1 = tf.get_variable("kernel")
    #         biases_dense1=tf.get_variable("bias")

    dense2 = tf.layers.dense(inputs=dense1_activated, units=4096, activation=None, name='dense2')

    dense2_activated = tf.nn.leaky_relu(dense2, alpha=0.1)

    #     with tf.variable_scope("dense2", reuse=True):
    #         weights_dense2 = tf.get_variable("kernel")
    #         biases_dense2=tf.get_variable("bias")

    # Add dropout operation; 0.6 probability that element will be kept
    #     dropout = tf.layers.dropout(
    #       inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(inputs=dense2_activated, units=1470, activation=None, name='logits')

    #     with tf.variable_scope("logits", reuse=True):
    #         weights_logits = tf.get_variable("kernel")
    #         biases_logits=tf.get_variable("bias")

    # sess = tf.Session()
    # #############check below st afterwards
    # sess.run(tf.global_variables_initializer())
    #     saver=tf.train.Saver({"yolo/conv_2/weights": weights_conv1, "yolo/conv_2/biases": biases_conv1,
    #                           "yolo/conv_4/weights": weights_conv2, "yolo/conv_4/biases": biases_conv2,
    #                          "yolo/conv_6/weights": weights_conv3, "yolo/conv_6/biases": biases_conv3,
    #                          "yolo/conv_7/weights": weights_conv4, "yolo/conv_7/biases": biases_conv4,
    #                          "yolo/conv_8/weights": weights_conv5, "yolo/conv_8/biases": biases_conv5,
    #                          "yolo/conv_9/weights": weights_conv6, "yolo/conv_9/biases": biases_conv6,
    #                          "yolo/conv_11/weights": weights_conv7, "yolo/conv_11/biases": biases_conv7,
    #                          "yolo/conv_12/weights": weights_conv8, "yolo/conv_12/biases": biases_conv8,
    #                          "yolo/conv_13/weights": weights_conv9, "yolo/conv_13/biases": biases_conv9,
    #                          "yolo/conv_14/weights": weights_conv10, "yolo/conv_14/biases": biases_conv10,
    #                          "yolo/conv_15/weights": weights_conv11, "yolo/conv_15/biases": biases_conv11,
    #                          "yolo/conv_16/weights": weights_conv12, "yolo/conv_16/biases": biases_conv12,
    #                          "yolo/conv_17/weights": weights_conv13, "yolo/conv_17/biases": biases_conv13,
    #                          "yolo/conv_18/weights": weights_conv14, "yolo/conv_18/biases": biases_conv14,
    #                          "yolo/conv_19/weights": weights_conv15, "yolo/conv_19/biases": biases_conv15,
    #                          "yolo/conv_20/weights": weights_conv16, "yolo/conv_20/biases": biases_conv16,
    #                          "yolo/conv_22/weights": weights_conv17, "yolo/conv_22/biases": biases_conv17,
    #                          "yolo/conv_23/weights": weights_conv18, "yolo/conv_23/biases": biases_conv18,
    #                          "yolo/conv_24/weights": weights_conv19, "yolo/conv_24/biases": biases_conv19,
    #                          "yolo/conv_25/weights": weights_conv20, "yolo/conv_25/biases": biases_conv20,
    #                          "yolo/conv_26/weights": weights_conv21, "yolo/conv_26/biases": biases_conv21,
    #                          "yolo/conv_28/weights": weights_conv22, "yolo/conv_28/biases": biases_conv22,
    #                          "yolo/conv_29/weights": weights_conv23, "yolo/conv_29/biases": biases_conv23,
    #                          "yolo/conv_30/weights": weights_conv24, "yolo/conv_30/biases": biases_conv24,
    #                          "yolo/fc_33/weights": weights_dense1, "yolo/fc_33/biases": biases_dense1,
    #                          "yolo/fc_34/weights": weights_dense2, "yolo/fc_34/biases": biases_dense2,
    #                          "yolo/fc_36/weights": weights_logits, "yolo/fc_36/biases": biases_logits})

    #     saver.restore(sess,"YOLO_small.ckpt")


    #     logits = tf.reshape(logits, [-1, 7, 7, 30])   SEE IF REQUIRED.

    #########################################################################

###########################################################
    if mode == tf.estimator.ModeKeys.PREDICT:
        pass
    #         spec = tf.estimator.EstimatorSpec(mode=mode,
    #                                           predictions=y_pred_cls)
    # Otherwise the estimator is supposed to be in either
    # training or evaluation-mode. Note that the loss-function
    # is also required in Evaluation mode.


    # This gives the LOSS for each image in the batch.
    loss = loss_fn.loss_fn(logits, labels)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # settings as per hizhang
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, decay_steps,
            decay_rate, staircase, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

        train_op = optimizer.minimize(
            loss=loss, global_step=global_step)

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=None)

        return spec

def normalize_labels(labels):
    bat_sz = labels.shape[0]
    for i in range(bat_sz):
        for j in range(7):
            for k in range(7)nmjjjjjjjjjjjjjjjjjjjjj                                                                                   67:
                for l in range(1, 5):
                    if l == 1 or l == 2:
                        labels[i, j, k, l] = (labels[i, j, k, l] % 64) / 64
                    elif l == 3 or l == 4:
                        labels[i, j, k, l] = labels[i, j, k, l] / 448

def main(unused_argv):
    # Load training and eval data
    pascal = pascal_voc.pascal_voc('train')
    images, labels = pascal.get()
    images = np.float64(images)
    labels = np.float64(labels)
    normalize_labels(labels)
    train_data = images  # Returns np.array
    train_labels = labels

    model = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                   model_dir="./checkpoints_yolo/")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=1,
        shuffle=True)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
    model.train(input_fn=train_input_fn)
    # writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    # writer.close()

    # #evaluation
    # eval_data = images  # Returns np.array
    # eval_labels = labels
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False)



if __name__ == "__main__":
    tf.app.run()
































