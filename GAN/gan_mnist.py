import os
import sys

sys.path.append(os.getcwd())

import time

import matplotlib

matplotlib.use('Agg')
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp'  # dcgan, wgan, or wgan-gp
DIM = 64  # Model dimensionality
BATCH_SIZE = 50  # Batch size
CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
ITERS = 200000  # How many generator iterations to train for
OUTPUT_DIM = 784  # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())
if not os.path.exists("mnist_output"):
    os.mkdir("mnist_output")


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 4 * DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4 * DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = output[:, :, :7, :7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2 * DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * DIM, 1, output)

    return tf.reshape(output, [-1])


real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += LAMBDA * gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.ones_like(disc_fake)
    ))

    disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake,
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real,
        labels=tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)


def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)),
        'mnist_output/samples_{}.png'.format(frame)
    )


# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)


def inf_train_gen():
    while True:
        for images, targets in train_gen():
            yield images


g_cost_sum = tf.summary.scalar("gen_cost", gen_cost)
d_cost_sum = tf.summary.scalar("disc_cost", disc_cost)

# self.writer.add_summary(summary_str, counter)
# Train loop
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./mnist_logs", session.graph)

    gen = inf_train_gen()

    for iteration in range(ITERS):
        start_time = time.time()

        _gen_cost = 0.
        if iteration > 0:
            _gen_cost,_, summary_str = session.run([gen_cost,gen_train_op, g_cost_sum])
            summary_writer.add_summary(summary_str, iteration)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = gen.next()
            _disc_cost, _, summary_str = session.run(
                [disc_cost, disc_train_op, d_cost_sum],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        summary_writer.add_summary(summary_str, iteration)
        # lib.plot.plot('train disc cost', _disc_cost)
        # lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        val_cost = 0.
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost,
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            val_cost = np.mean(dev_disc_costs)
            # lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            print("iter: ", iteration, 'train_cost', _gen_cost,"val_cost: ", val_cost)
            # lib.plot.flush()

        # lib.plot.tick()
