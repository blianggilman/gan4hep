"""
MLP-based GANs
"""
from types import SimpleNamespace
import functools
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt


def sum_trainables(module):
    return sum([tf.size(v) for v in module.trainable_variables])

class GANBase(snt.Module):
    def __init__(self, noise_dim, batch_size, name=None):
        super().__init__(name=name)
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.generator = None
        self.discriminator = None

    def get_noise_batch(self):
        noise_shape = [self.batch_size, self.noise_dim]
        #return tf.random.normal(noise_shape, dtype=tf.float32)
        return tf.random.uniform(noise_shape, minval=-1, maxval=1, seed=None)

    def create_ganenerator_inputs(self, cond_inputs=None):
        inputs = self.get_noise_batch()
        if cond_inputs is not None:
            inputs = tf.concat([cond_inputs, inputs], axis=-1)
        return inputs

    def generate(self, cond_inputs=None, is_training=True):
        inputs = self.create_ganenerator_inputs(cond_inputs)
        output = self.generator(inputs, is_training)
        if cond_inputs is not None:
            output = tf.concat([cond_inputs, output], axis=-1)
        return output

    def discriminate(self, inputs, is_training=True):
        return self.discriminator(inputs, is_training)

    def num_trainable_vars(self):
        return sum_trainables(self.generator), sum_trainables(self.discriminator)


def Discriminator_Regularizer(p_true, grad_D_true_logits, p_gen, grad_D_gen_logits):
    """
    Args:
        p_true: probablity from Discriminator for true events
        grad_D_true_logits: gradient of Discrimantor logits w.r.t its input variables
        p_gen: probability from Discriminator for generated events
        grad_D_gen_logits: gradient of Discrimantor logits w.r.t its input variables
    Returns:
        discriminator regularizer
    """
    # grad_D_true_logits_norm = tf.norm(
    #     tf.reshape(grad_D_true_logits, [batch_size, -1]),
    #     axis=1, keepdims=True
    # )
    # grad_D_gen_logits_norm = tf.norm(
    #     tf.reshape(grad_D_gen_logits, [batch_size, -1]),
    #     axis=1, keepdims=True
    # )
    grad_D_true_logits_norm = tf.norm(grad_D_true_logits, axis=1, keepdims=True)
    grad_D_gen_logits_norm  = tf.norm(grad_D_gen_logits,  axis=1, keepdims=True)

    assert grad_D_true_logits_norm.shape == p_true.shape, "{} {}".format(grad_D_true_logits_norm.shape, p_true.shape)
    assert grad_D_gen_logits_norm.shape == p_gen.shape, "{} {}".format(grad_D_gen_logits_norm.shape, p_gen.shape)
        
    reg_true = tf.multiply(tf.square(1.0 - p_true), tf.square(grad_D_true_logits_norm))
    reg_gen = tf.multiply(tf.square(p_gen), tf.square(grad_D_gen_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_true + reg_gen)
    return disc_regularizer, grad_D_true_logits_norm, grad_D_gen_logits_norm, reg_true, reg_gen


class GANOptimizer(snt.Module):

    def __init__(self,
                gan, 
                disc_lr=2e-4,
                gen_lr=5e-5,
                num_epochs=100,
                decay_disc_lr=True,
                decay_gen_lr=True,
                decay_epochs=2,
                decay_base=0.96,
                loss_type='logloss',
                gamma_reg=1e-3,
                debug=False,
                name=None, *args, **kwargs):
        super().__init__(name=name)
        self.gan = gan
        self.hyparams = SimpleNamespace(
            disc_lr=disc_lr,
            gen_lr=gen_lr,
            num_epochs=num_epochs,
            with_disc_reg=True,
            loss_type=loss_type,
            decay_disc_lr=decay_disc_lr,
            decay_gen_lr=decay_gen_lr,
            gamma_reg=gamma_reg
        )
        self.debug = debug

        # learning rate decay
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/exponential_decay
        self.decay_epochs = tf.constant(decay_epochs, dtype=tf.float32) # decay steps
        self.decay_base = tf.constant(decay_base, dtype=tf.float32)     # decay base

        self.disc_lr = tf.Variable(disc_lr, trainable=False, name='disc_lr', dtype=tf.float32)
        self.gen_lr = tf.Variable(gen_lr, trainable=False, name='gen_lr', dtype=tf.float32)

        # regularization term decays
        self.gamma_reg = tf.Variable(gamma_reg, trainable=False, name="gamma_reg", dtype=tf.float32)

        # two different optimizers        
        self.disc_opt = snt.optimizers.Adam(learning_rate=self.hyparams.disc_lr, beta1=0.5, beta2=0.9)
        self.gen_opt = snt.optimizers.Adam(learning_rate=self.hyparams.gen_lr, beta1=0.5, beta2=0.9)

        self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)

        self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits \
            if loss_type == 'logloss' else tf.compat.v1.losses.mean_squared_error


    def disc_step(self, truth_inputs, cond_inputs=None, lr_mult=1.0):
        gan = self.gan
        if self.debug and cond_inputs is not None:
            print("cond inputs:", cond_inputs.shape)

        # shuffle the inputs before feeding discriminator
        # gen_evts = gan.generate(cond_inputs)
        # all_inputs = tf.concat([gen_evts, truth_inputs], axis=0)
        # all_truths = tf.concat([tf.zeros_like(gen_evts),
        #                     tf.ones_like(truth_inputs)], axis=0)
        # indices = tf.range(start=0, limit=tf.shape(all_inputs)[0], dtype=tf.int32)
        # shuffled_idx = tf.random.shuffle(indices)
        # shuffled_inputs = tf.gather(all_inputs, shuffled_idx)
        # shuffled_truths = tf.gather(all_truths, shuffled_idx)

        with tf.GradientTape() as tape, tf.GradientTape() as true_tape, tf.GradientTape() as fake_tape:
            gen_evts = gan.generate(cond_inputs)

            true_tape.watch(truth_inputs)
            fake_tape.watch(gen_evts)
            real_output = gan.discriminate(truth_inputs)
            fake_output = gan.discriminate(gen_evts)
            if self.debug:
                print("generated info:", gen_evts.shape, gen_evts[0])
                print("discriminator inputs:", truth_inputs.shape, truth_inputs[0])
                print("discriminator info:", real_output.shape, real_output[0])


            # loss = tf.reduce_mean(self.loss_fn(tf.ones_like(real_output), real_output) \
            #                     + self.loss_fn(tf.zeros_like(fake_output), fake_output))

            loss = tf.reduce_mean(self.loss_fn(tf.zeros_like(fake_output), fake_output) \
                                + self.loss_fn(tf.ones_like(real_output), real_output))


            if self.hyparams.with_disc_reg:
                grad_logits_true = true_tape.gradient(real_output, truth_inputs)
                grad_logits_gen = fake_tape.gradient(fake_output, gen_evts)

                real_scores = tf.sigmoid(real_output) if self.hyparams.loss_type == "logloss"\
                    else real_output
                fake_scores = tf.sigmoid(fake_output) if self.hyparams.loss_type == "logloss"\
                    else fake_output

                regularizers = Discriminator_Regularizer(
                    real_scores,
                    grad_logits_true,
                    fake_scores,
                    grad_logits_gen
                )
                reg_loss = regularizers[0]
                assert reg_loss.shape == loss.shape
                self.gamma_reg.assign(self.hyparams.gamma_reg*lr_mult)
                loss += self.gamma_reg*reg_loss

        disc_params = gan.discriminator.trainable_variables
        disc_grads = tape.gradient(loss, disc_params)

        if self.hyparams.decay_disc_lr:
            self.disc_lr.assign(self.hyparams.disc_lr * lr_mult)

        self.disc_opt.apply(disc_grads, disc_params)
        if self.hyparams.with_disc_reg:
            return loss, *regularizers
        else:
            return loss,


    def gen_step(self, cond_inputs=None, lr_mult=1.0):
        gan = self.gan

        with tf.GradientTape() as tape:
            gen_graph = gan.generate(cond_inputs)
            fake_output = gan.discriminate(gen_graph)
            loss = tf.reduce_mean(self.loss_fn(tf.ones_like(fake_output), fake_output))

        gen_params = gan.generator.trainable_variables
        gen_grads = tape.gradient(loss, gen_params)

        if self.hyparams.decay_gen_lr:
            self.gen_lr.assign(self.hyparams.gen_lr * lr_mult)

        self.gen_opt.apply(gen_grads, gen_params)
        return loss

    def _get_lr_mult(self, epoch):
        # exponential decay the learning rate...
        epoch = tf.cast(epoch, tf.float32)
        return tf.math.pow(self.decay_base, epoch/self.decay_epochs)


    def step(self, targets_tr, epoch, inputs_tr):
        lr_mult = self._get_lr_mult(epoch)
        disc_loss = self.disc_step(targets_tr, inputs_tr, lr_mult=lr_mult)
        gen_loss = self.gen_step(inputs_tr, lr_mult=lr_mult)
        return disc_loss, gen_loss, lr_mult
