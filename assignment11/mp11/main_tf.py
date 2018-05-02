"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=64,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
        
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.normal(size = [batch_size,model._nlatent])
        
        for i in range(4):
            model.session.run(model.discriminator_train_op, feed_dict={model.x_placeholder: batch_x,  model.z_placeholder: batch_z, model.learning_rate: learning_rate})
        model.session.run(model.generator_train_op, feed_dict={model.z_placeholder: batch_z, model.learning_rate: learning_rate})
        
        # Print loss info
        if(step % 500 == 0):
            print("_______________")
            print("Step: " , step)
            print("Discriminator loss: " , model.session.run(model.d_loss, feed_dict={model.x_placeholder: batch_x,  model.z_placeholder: batch_z}))
            print("Generator loss: " , model.session.run(model.g_loss, feed_dict={model.z_placeholder: batch_z}))
    
    out = np.empty((28*20, 28*20))
    for i in range(20):
        for j in range(20):
            z_input = np.random.normal(size = [1,model._nlatent])
            img = model.session.run(model.x_hat, feed_dict={model.z_placeholder: z_input})
            out[i*28:(i+1)*28,
                j*28:(j+1)*28] = img[0].reshape(28, 28)
    plt.imsave('generator_image.png', out, cmap="gray")   
    
def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan(nlatent = 128)

    # Start training
    train(model, mnist_dataset)


if __name__ == "__main__":
    tf.app.run()
