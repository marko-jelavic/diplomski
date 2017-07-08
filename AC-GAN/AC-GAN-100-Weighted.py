import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# *** GENERATING LATENT VARIABLES ***

def leakyRelu(x):
    return tf.maximum(x, 0.2*x)

def LatentVariables(batch_size):
    return np.random.uniform(-1, 1, size=(batch_size, 100))

def LoadingEvenlyDistributedMnist(n_per_class):

    begining_distribution = np.zeros(10)
    goal_distribution = np.ones(10)*n_per_class

    x_batch = []
    y_batch = []
    temp_x, temp_y = mnist.train.images, mnist.train.labels
    for i in range(temp_x.shape[0]):
        for j in range(temp_y[i].shape[0]):
            if temp_y[i][j] == 1.0 and begining_distribution[j] < n_per_class:
                x_batch.append(temp_x[i])
                y_batch.append(temp_y[i])
                begining_distribution[j]+=1
    
        if np.array_equal(goal_distribution, begining_distribution):
            break

    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)
    return x_batch, y_batch

# *** PLOTTING SAMPLES ***

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# *** GENERATOR ARCHITECTURE *** 

y = tf.placeholder(tf.float32, shape=(None, 10))

z = tf.placeholder(tf.float32, shape=(None, 100))

with tf.variable_scope("Generator"):
    
    g_w1 = tf.Variable(tf.truncated_normal([110, 1024], stddev = 0.02))
    g_b1 = tf.constant(0.1, shape=[1024])

    g_w2 = tf.Variable(tf.truncated_normal([1024, 6272], stddev = 0.02))
    g_b2 = tf.constant(0.1, shape=[6272])

    g_conv2 = tf.Variable(tf.truncated_normal([7, 7, 128, 128], stddev = 0.02))
    g_conv2_bias = tf.constant(0.1, shape=[128])

    g_conv3 = tf.Variable(tf.truncated_normal([14, 14, 1, 128], stddev = 0.02))
    g_conv3_bias = tf.constant(0.1, shape=[1])


def Generator(z, y):

    batch_size = tf.shape(z)[0]

    inputs = tf.concat(axis=1, values=[z, y])

    with tf.variable_scope("G1"):

        g1 = tf.matmul(inputs, g_w1) + g_b1
        g1_batch_norm = tf.contrib.layers.batch_norm(g1, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Generator")
        g1 = tf.nn.relu(g1_batch_norm)

    with tf.variable_scope("G2"):

        g2 = tf.matmul(g1, g_w2) + g_b2
        g2_batch_norm = tf.contrib.layers.batch_norm(g2, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Generator")
        g2_relu = tf.nn.relu(g2_batch_norm)
        g2 = tf.reshape(g2_relu, [batch_size, 7, 7, 128])

    with tf.variable_scope("G3"):

        g3 = tf.nn.conv2d_transpose(g2, g_conv2, [batch_size, 14, 14, 128], [1, 2, 2, 1], padding="SAME")
        g3 = tf.nn.bias_add(g3, g_conv2_bias)
        g3_batch_norm = tf.contrib.layers.batch_norm(g3, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Generator")
        g3 = tf.nn.relu(g3_batch_norm)

    with tf.variable_scope("G4"):

        g4 = tf.nn.conv2d_transpose(g3, g_conv3, [batch_size, 28, 28, 1], [1, 2, 2, 1], padding="SAME")
        g4 = tf.nn.bias_add(g4, g_conv3_bias)
    
    g4_reshaped = tf.reshape(g4, [-1, 784])

    G = tf.nn.sigmoid(g4_reshaped)

    return G

# *** DISCRIMINATOR ARCHITECTURE ***

x = tf.placeholder(tf.float32, shape=(None, 784))

with tf.variable_scope("Discriminator"):

    d_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev = 0.02))
    d_bias1 = tf.constant(0.1, shape=[16])

    d_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.02))
    d_bias2 = tf.constant(0.1, shape=[32])

    d_conv3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.02))
    d_bias3 = tf.constant(0.1, shape=[64])

    d_conv4 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev = 0.02))
    d_bias4 = tf.constant(0.1, shape=[128])

    d_w_sample = tf.Variable(tf.truncated_normal([512, 1], stddev = 0.02))
    d_b_sample = tf.constant(0.1, shape=[1])

    d_w_class = tf.Variable(tf.truncated_normal([512, 10], stddev = 0.02))
    d_b_class = tf.constant(0.1, shape=[10])


def Discriminator(x):

    batch_size = tf.shape(z)[0]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("D1"):

        d1 = tf.nn.conv2d(x_image, d_conv1, strides=[1, 2, 2, 1], padding="SAME") + d_bias1
        d1_batch_norm = tf.contrib.layers.batch_norm(d1, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Discriminator")
        d1 = leakyRelu(d1_batch_norm)

    with tf.variable_scope("D2"):

        d2 = tf.nn.conv2d(d1, d_conv2, strides=[1, 2, 2, 1], padding="SAME") + d_bias2
        d2_batch_norm = tf.contrib.layers.batch_norm(d2, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Discriminator")
        d2 = leakyRelu(d2_batch_norm)

    with tf.variable_scope("D3"):

        d3 = tf.nn.conv2d(d2, d_conv3, strides=[1, 2, 2, 1], padding="SAME") + d_bias3
        d3_batch_norm = tf.contrib.layers.batch_norm(d3, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Discriminator")
        d3 = leakyRelu(d3_batch_norm)

    with tf.variable_scope("D4"):

        d4 = tf.nn.conv2d(d3, d_conv4, strides=[1, 2, 2, 1], padding="SAME") + d_bias4
        d4_batch_norm = tf.contrib.layers.batch_norm(d4, decay=0.9, updates_collections=None, epsilon=0.00001, scale=True, is_training=True, scope="Discriminator")
        d4 = leakyRelu(d4_batch_norm)

    with tf.variable_scope("D5"):

        d4 = tf.reshape(d4, [-1, 512])

        D_sample_logit = tf.matmul(d4, d_w_sample) + d_b_sample
        #D_sample = tf.nn.sigmoid(D_sample_logit)

        D_class_logit = tf.matmul(d4, d_w_class) + d_b_class
        D_class = tf.nn.softmax(D_class_logit)
        

    return D_sample_logit, D_class_logit, D_class
        
        
generator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")


G = Generator(z, y)


with tf.variable_scope("Real"):

    D_real_sample, D_real_class, D_real_class = Discriminator(x)

with tf.variable_scope("Generated"):

    D_generated_sample, D_generated_class, D_sample_class = Discriminator(G)


Classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_real_class, labels=y)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_generated_class, labels=y))

D_sample_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_sample, labels=tf.ones_like(D_real_sample)))

D_sample_loss_generated = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_generated_sample, labels=tf.zeros_like(D_generated_sample)))

D_sample_loss = D_sample_loss_real + D_sample_loss_generated

Discriminator_loss = 0.9*Classifier_loss + 0.1*D_sample_loss

G_sample_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_generated_sample, labels=tf.ones_like(D_generated_sample)))

Generator_loss = Classifier_loss + G_sample_loss

train_D = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(Discriminator_loss, var_list = discriminator_params)
train_G = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(Generator_loss, var_list = generator_params)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

num_iters = 100001
batch_size = 1000

if not os.path.exists('generated_samples/'):
    os.makedirs('generated_samples/')

it = 0


x_samples, y_samples = LoadingEvenlyDistributedMnist(100)

for i in range(num_iters):

    #x_samples, y_samples = mnist.train.next_batch(batch_size)
    z_samples = LatentVariables(batch_size)
    _, error2 = sess.run([train_D, Discriminator_loss], {x: x_samples, z: z_samples, y: y_samples})
    _, error4 = sess.run([train_G, Generator_loss], {x: x_samples, z: z_samples, y: y_samples})

    if i % 5000 == 0:
        print ("Iteration: " + str(i))
        print ("Discriminator loss " + str(error2))
        print ("Generator loss " + str(error4))

        samples = sess.run(G, feed_dict={z: z_samples[:25], y: y_samples[:25]})

        fig = plot(samples)
        plt.savefig
        plt.savefig('generated_samples/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        it += 1
        plt.close(fig)

x_test, y_test = mnist.test.images, mnist.test.labels
y_predicted_test = sess.run([D_real_class], {x: x_test})[0]
y_test_notonehot = []
y_predicted_notonehot = []
for i in range(len(y_test)):
    y_test_notonehot.append(np.argmax(y_test[i]))
    y_predicted_notonehot.append(np.argmax(y_predicted_test[i]))

accuracy = accuracy_score(y_test_notonehot, y_predicted_notonehot)
print ("Test accuracy on entire MNIST: " + str(accuracy))
