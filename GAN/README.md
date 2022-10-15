Porting to Tensorflow 2 borrowed
from [Link](https://github.com/marload/GANs-TensorFlow2/blob/77d851846b7c3675622aca39f4c59314980f2f41/GAN/GAN.py)

A bit puzzled why the examples look worse with the TF2 implementation, tried to keep parameters the same.

**UPDATE:** After switching latent variable generation from np.random.normal to tf.random.uniform we get closer results
to initial implementation, no smudgeness.

Relevant papers [Paper1](https://arxiv.org/pdf/1406.2661.pdf) [Paper2](https://arxiv.org/pdf/1606.03498.pdf)