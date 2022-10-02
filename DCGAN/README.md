Porting to Tensorflow 2 borrowed
from [Link](https://github.com/marload/GANs-TensorFlow2/blob/77d851846b7c3675622aca39f4c59314980f2f41/GAN/GAN.py)

A bit puzzled why the examples look worse with the TF2 implementation, tried to keep parameters the same.

**UPDATE:** After switching latent variable generation from np.random.normal to tf.random.uniform we get closer results
to initial implementation. Sometimes we cutting out initialisation parts helps, sometimes not.
GAN with 3 fully connected layers experiences some fading later on in the iterations too.