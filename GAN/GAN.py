import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import models, layers

(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

BATCH_SIZE = 500
NUM_STEPS = 150001
IMG_SHAPE = (28, 28, 1)
Z_DIM = 100


# Generating latent variables


def generate_latent_variables(batch=BATCH_SIZE):
    return tf.random.uniform([batch, Z_DIM], minval=-1, maxval=1.0)


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
        plt.imshow(sample, cmap='Greys_r')

    return fig


# *** GENERATOR ARCHITECTURE ***

generator = models.Sequential()
generator.add(layers.Dense(200, activation="relu", input_shape=(100,),
                           kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                           bias_initializer=tf.keras.initializers.truncated_normal(stddev=0.01)))
generator.add(
    layers.Dense(784, activation="sigmoid", kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                 bias_initializer=tf.keras.initializers.truncated_normal(stddev=0.01)))
generator.add(layers.Reshape(IMG_SHAPE))


def gen_loss_fn(fake_logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_logits), fake_logits))


discriminator = models.Sequential()
discriminator.add(layers.InputLayer(IMG_SHAPE))
discriminator.add(layers.Flatten())
discriminator.add(
    layers.Dense(200, activation="relu", kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                 bias_initializer=tf.keras.initializers.truncated_normal(stddev=0.01)))
discriminator.add(
    layers.Dense(1, kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.01),
                 bias_initializer=tf.keras.initializers.truncated_normal(stddev=0.01)))


def disc_loss_fn(real_logits, fake_logits):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_logits), real_logits))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_logits), fake_logits))
    return real_loss + fake_loss


train_images = train_images / 255
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE, drop_remainder=True).shuffle(60000).repeat())

g_optim = tf.keras.optimizers.Adam()
d_optim = tf.keras.optimizers.SGD(0.01)


@tf.function
def train_step(images):
    z = generate_latent_variables()
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = generator(z, training=True)

        fake_logits = discriminator(fake_images, training=True)
        real_logits = discriminator(images, training=True)

        d_loss = disc_loss_fn(real_logits, fake_logits)
        g_loss = gen_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return g_loss, d_loss


def train(ds):
    ds = iter(ds)
    it = 0
    for step in range(NUM_STEPS):
        images = next(ds)
        train_step(images)

        if step % 5000 == 0:
            gif = generate_latent_variables(25)
            print("[{}/{}]".format(step, NUM_STEPS))
            fig = plot(generator(gif))
            plt.savefig('new_samples/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
            plt.close(fig)
            it += 1


train(train_ds)
