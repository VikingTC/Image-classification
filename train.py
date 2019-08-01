import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
import matplotlib.image as image
from PIL import Image
import os
from collections import defaultdict
height = 299
width = 299
channels = 3
X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='X')
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_ponits = inception.inception_v3(X, num_classes=1001, is_training=False)
predictions = end_ponits['Predictions']
inception_saver  = tf.train.Saver()

# find the point in the graph where we should attach the new output
# layer.( it should be the layer right before the current output layer
prelogits = tf.squeeze(end_ponits["PreLogits"], axis=[1, 2])
n_ouptus = 5
with tf.name_scope('new_ouput_layer'):
    flower_logits = tf.layers.dense(prelogits, n_ouptus, name='flower_logits')
    y_proba = tf.nn.softmax(flower_logits, name='y_proba')

y = tf.placeholder(tf.int32, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')
with tf.name_scope('train'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=flower_logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.MomentumOptimizer(momentum=0.95)
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flower_logits')
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_saver'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

flowers_root_path = os.path.join('datasets', 'flowers aug')
flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
image_paths = defaultdict(list)
for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith('.jpg'):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

for paths in image_paths.values():
    paths.sort()

# represent the classes as int
flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
flower_paths_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_classes.append((path, flower_class_ids[flower_class]))

test_ratio = 0.2
train_size = int(len(flower_paths_classes) * (1 - test_ratio))

np.random.shuffle(flower_paths_classes)

flower_paths_and_classes_train = flower_paths_classes[:train_size]
flower_paths_and_classes_test = flower_paths_classes[train_size:]

# preprocess a set of image
from random import sample

def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [image.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    X_batch = 2 * np.stack(images) - 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch

X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size=4)
X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))
n_epochs = 100
batch_size = 40
n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, 'inception_v3.ckpt')

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("  Last batch accuracy:", acc_batch)

        save_path = saver.save(sess, "./my_flowers_model")

n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)