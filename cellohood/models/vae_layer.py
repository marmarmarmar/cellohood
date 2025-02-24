from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as k_backend
from keras import layers as k_layers
from tensorflow_probability import distributions as tfd

DEFAULT_OUTPUT_SIZE = 40


VAELayerOutput = namedtuple(
    'VAELayerOutput',
    [
        'sample',
        'kl',
        'sample_log_prior',
        'sample_log_posterior',
    ]
)


class VAELayer(k_layers.Layer):

    def __init__(
            self,
            latent_size=64,
    ):
        super(VAELayer, self).__init__()
        self.latent_size = latent_size

    def call(self, x):
        posterior_loc, posterior_scale_diag = x
        reference_normal = tfd.MultivariateNormalDiag(
            loc=k_backend.zeros(shape=(self.latent_size,)),
            scale_diag=k_backend.ones(shape=(self.latent_size,))
        )
        posterior_normal = tfd.MultivariateNormalDiag(
            loc=posterior_loc,
            scale_diag=posterior_scale_diag,
        )
        sample = posterior_normal.sample()
        kl = posterior_normal.kl_divergence(reference_normal)
        sample_log_prior = reference_normal.log_prob(sample)
        sample_log_posterior = posterior_normal.log_prob(sample)
        return VAELayerOutput(
            sample=sample,
            kl=kl,
            sample_log_prior=sample_log_prior,
            sample_log_posterior=sample_log_posterior,
        )


VAEGMMLayerOutput = namedtuple(
    'VAEGMMLayerOutput',
    [
        'sample',
        'sample_log_prior',
        'sample_log_posterior',
        'sample_mixture_posterior',
    ]
)


VAEGMMLayerOutputWithCondition = namedtuple(
    'VAEGMMLayerOutput',
    [
        'sample',
        'sample_log_prior',
        'sample_log_posterior',
    ]
)


class VAEGMMLayer(k_layers.Layer):

    def __init__(
            self,
            latent_size=64,
            nb_components=DEFAULT_OUTPUT_SIZE,
            components_scale=0.3,
            starting_components_scale=1.0,
    ):
        super(VAEGMMLayer, self).__init__()
        self.latent_size = latent_size
        self.nb_components = nb_components
        self.components_scale = components_scale

        self.components_logits = k_backend.variable(
            np.zeros((self.nb_components,)),
            name='gmm_logits'
        )
        self.component_means = k_backend.variable(
            np.random.normal(size=(self.nb_components, self.latent_size)) * starting_components_scale
        )
        self.component_diags = k_backend.constant(
            np.ones(shape=(self.nb_components, self.latent_size)) * self.components_scale
        )
        self.mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=self.components_logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.component_means,
                scale_diag=self.component_diags,
            )
        )

    def _per_mixture_component_log_prob(self, x):
        x = self.mixture._pad_sample_dims(x)
        log_prob_x = self.mixture.components_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = tf.math.log_softmax(
            self.mixture.mixture_distribution.logits, axis=-1)  # [B, k]
        return log_prob_x + log_mix_prob  # [S, B, k]

    def call(self, x):
        posterior_loc, posterior_scale_diag = x
        posterior_normal = tfd.MultivariateNormalDiag(
            loc=posterior_loc,
            scale_diag=posterior_scale_diag,
        )
        sample = posterior_normal.sample()

        prior_likelihood = self.mixture.log_prob(sample)
        posterior_likelihood = posterior_normal.log_prob(sample)

        posterior_entropy = posterior_normal.entropy()
        categorical_posterior = self._per_mixture_component_log_prob(sample)
        return VAEGMMLayerOutput(
            sample=sample,
            sample_log_prior=prior_likelihood,
            sample_log_posterior=posterior_likelihood,
            sample_mixture_posterior=categorical_posterior,
        )


class VAEGMMLayerWithCondition(k_layers.Layer):

    def __init__(
            self,
            latent_size=64,
            nb_components=DEFAULT_OUTPUT_SIZE,
            components_scale=0.3,
            starting_components_scale=1.0,
    ):
        super(VAEGMMLayerWithCondition, self).__init__()
        self.latent_size = latent_size
        self.nb_components = nb_components
        self.components_scale = components_scale

        self.component_means = k_backend.variable(
            np.random.normal(size=(self.nb_components, self.latent_size)) * starting_components_scale
        )
        self.component_diags = k_backend.constant(
            np.ones(shape=(self.nb_components, self.latent_size)) * self.components_scale
        )

    def call(self, x):
        posterior_loc, posterior_scale_diag, condition = x

        posterior_normal = tfd.MultivariateNormalDiag(
            loc=posterior_loc,
            scale_diag=posterior_scale_diag,
        )
        sample = posterior_normal.sample()
        posterior_likelihood = posterior_normal.log_prob(sample)

        post_cond_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=condition),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.component_means,
                scale_diag=self.component_diags,
            )
        )
        prior_likelihood = post_cond_mixture.log_prob(sample)
        posterior_entropy = posterior_normal.entropy()

        return VAEGMMLayerOutputWithCondition(
            sample=sample,
            sample_log_prior=prior_likelihood,
            sample_log_posterior=posterior_likelihood,
        )


class HierarchicalVAEGMMLayer(k_layers.Layer):

    def __init__(
            self,
            latent_size=64,
            nb_of_components=50,
            nb_of_out_components=DEFAULT_OUTPUT_SIZE,
            components_scale=0.3,
            starting_components_scale=1.0,
    ):
        super(HierarchicalVAEGMMLayer, self).__init__()
        self.latent_size = latent_size
        self.nb_components = nb_of_components
        self.nb_of_out_components = nb_of_out_components
        self.components_scale = components_scale

        self.components_logits = k_backend.variable(
            np.zeros((self.nb_components,)),
            name='gmm_logits'
        )
        self.components_out_logits = k_backend.variable(
            np.zeros((self.nb_components, nb_of_out_components)),
            name='gmm_logits'
        )
        self.component_means = k_backend.variable(
            np.random.normal(size=(self.nb_components, self.latent_size)) * starting_components_scale
        )
        self.component_diags = k_backend.constant(
            np.ones(shape=(self.nb_components, self.latent_size)) * self.components_scale
        )
        self.mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=self.components_logits),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.component_means,
                scale_diag=self.component_diags,
            )
        )

    def _per_mixture_component_log_prob(self, x):
        x = self.mixture._pad_sample_dims(x)
        log_prob_x = self.mixture.components_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = tf.math.log_softmax(
            self.mixture.mixture_distribution.logits, axis=-1)  # [B, k]
        return log_prob_x + log_mix_prob  # [S, B, k]

    def call(self, x):
        posterior_loc, posterior_scale_diag = x
        posterior_normal = tfd.MultivariateNormalDiag(
            loc=posterior_loc,
            scale_diag=posterior_scale_diag,
        )
        sample = posterior_normal.sample()

        prior_likelihood = self.mixture.log_prob(sample)
        posterior_likelihood = posterior_normal.log_prob(sample)

        posterior_entropy = posterior_normal.entropy()
        categorical_posterior = self._per_mixture_component_log_prob(sample)

        posterior_logit_mixture = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                logits=categorical_posterior),
            components_distribution=tfd.Categorical(
                logits=self.components_out_logits,
            )
        )

        outs = tf.concat([
            tf.reshape(
                posterior_logit_mixture.log_prob(i),
                (-1, 1),
            )
            for i in range(self.nb_of_out_components)
        ], axis=-1)
        print(outs.shape)

        return VAEGMMLayerOutput(
            sample=sample,
            sample_log_prior=prior_likelihood,
            sample_log_posterior=posterior_likelihood,
            sample_mixture_posterior=outs,
        )
