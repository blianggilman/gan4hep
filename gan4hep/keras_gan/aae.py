"""
Adversarial Autoencoders
https://arxiv.org/abs/1511.05644
"""
import numpy as np
import os


import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

import tqdm


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output, reconstructed_evts, truth_evts):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))*0.001  \
        + tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(truth_evts, reconstructed_evts))*0.999


class AAE():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 4, disable_tqdm=False, lr=0.0001):
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim
        self.disable_tqdm = disable_tqdm

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # ============
        # Optimizers
        # ============
        self.generator_optimizer = keras.optimizers.Adam(lr)
        self.discriminator_optimizer = keras.optimizers.Adam(lr)

        # Build the critic
        self.discriminator = self.build_critic()
        self.discriminator.summary()

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.encoder.summary()

        self.decoder = self.build_decoder()
        self.decoder.summary()

    def build_encoder(self):
        # <NOTE, it's common practice to avoid using batch normalization when training VAE,
        # I assume it is also the case for AAE?
        # <https://www.tensorflow.org/tutorials/generative/cvae#network_architecture>
        """
        Input: conditional input + output info
        """
        model = keras.Sequential([
            keras.Input(shape=(self.gen_output_dim,)),
            layers.Dense(256, activation='relu'),            
            layers.Dense(256, activation='relu'),
            layers.Dense(self.noise_dim + self.noise_dim)
        ], name='Encoder')

        return model

    def encode(self, x, training=False):
        mean, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def build_decoder(self):
        model = keras.Sequential([
            keras.Input(shape=(self.gen_input_dim,)),
            layers.Dense(256, activation='relu'),            
            layers.Dense(256, activation='relu'),
            layers.Dense(self.gen_output_dim, activation='tanh'),
        ], name='Decoder')
        return model


    def build_critic(self):
        gen_output_dim = self.noise_dim + self.cond_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_output_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Dense(1, activation='sigmoid'),
        ], name='Discriminator')
        return model


    def train(self, train_truth, epochs, batch_size, test_truth, log_dir, evaluate_samples_fn,
        train_in=None, test_in=None):
        # ======================================
        # construct testing data once for all
        # ======================================
        AUTO = tf.data.experimental.AUTOTUNE
        noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], self.noise_dim))
        test_in = np.concatenate(
            [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise


        testing_data = tf.data.Dataset.from_tensor_slices(
            (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

        # ====================================
        # Checkpoints and model summary
        # ====================================
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        checkpoint = tf.train.Checkpoint(
            encoder=self.encoder,
            decoder=self.decoder,
            discriminator=self.discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
        logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

        summary_dir = os.path.join(log_dir, "logs")
        summary_writer = tf.summary.create_file_writer(summary_dir)

        img_dir = os.path.join(log_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)

        @tf.function
        def train_step(gen_in_4vec, cond_in, truth_4vec):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # encoder
                latent_real = gen_in_4vec
                mean, logvar = self.encode(truth_4vec, training=True)
                latent_fake = self.reparameterize(mean, logvar)
                latent_fake = tf.concat([cond_in, latent_fake], axis=-1)

                # discriminator
                real_output = self.discriminator(latent_real, training=True)
                fake_output = self.discriminator(latent_fake, training=True)
                disc_loss = discriminator_loss(real_output, fake_output)

                # train encoder and decoder
                replica_evts = self.decoder(latent_fake, training=True)
                gen_loss = generator_loss(latent_fake, replica_evts, truth_4vec)
                
            autoencoder_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients_of_generator = gen_tape.gradient(gen_loss, autoencoder_vars)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, autoencoder_vars))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return disc_loss, gen_loss

        best_wdis = 9999
        best_epoch = -1
        with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
            for epoch in t0:

                # compose the training dataset by generating different noises for each epochs
                noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
                train_inputs = np.concatenate(
                    [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise


                dataset = tf.data.Dataset.from_tensor_slices(
                    (train_inputs, train_in, train_truth)).shuffle(2*batch_size).batch(batch_size, drop_remainder=True).prefetch(AUTO)

                tot_loss = []
                for data_batch in dataset:
                    tot_loss.append(list(train_step(*data_batch)))

                tot_loss = np.array(tot_loss)
                avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
                loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])

                tot_wdis = evaluate_samples_fn(self.decoder, epoch, testing_data, summary_writer, img_dir, **loss_dict)
                if tot_wdis < best_wdis:
                    ckpt_manager.save()
                    best_wdis = tot_wdis
                    best_epoch = epoch
                t0.set_postfix(**loss_dict, BestD=best_wdis, BestE=best_epoch, CurrentD=tot_wdis)
        tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
        logging.info(tmp_res)
        summary_logfile = os.path.join(summary_dir, 'results.txt')
        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='ERROR',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)


    from gan4hep.utils_gan import generate_and_save_images
    from gan4hep.preprocess import herwig_angles

    train_in, train_truth, test_in, test_truth = herwig_angles(args.filename, args.max_evts)

    batch_size = args.batch_size
    gan = AAE()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, args.log_dir,
        generate_and_save_images,
        train_in, test_in
    )