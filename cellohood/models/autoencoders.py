from keras import backend as k_backend
from keras import layers as k_layers
from keras import losses as k_losses
from keras import models as k_models
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from cellohood.models import encoders
from cellohood.models import decoders
from cellohood.models import vae_layer


DEFAULT_NB_OF_CELL_TYPES = 16


class AutoEncoder(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        base_input = inputs
        latent = self.encoder(base_input)
        return self.decoder(latent)


class BaseMarkerAutoEncoder(AutoEncoder):

    def __init__(self):
        super(BaseMarkerAutoEncoder, self).__init__(
            encoder=encoders.BaseEncoder(),
            decoder=decoders.BaseDecoder(),
        )


class BaseCellAutoEncoder(AutoEncoder):

    def __init__(self):
        super(BaseCellAutoEncoder, self).__init__(
            encoder=encoders.BaseEncoder(),
            decoder=decoders.BaseDecoder(),
        )


class VAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            vae_layer,
    ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae_layer = vae_layer

    def call(self, inputs):
        base_input = inputs
        latent_mean, latent_scale = self.encoder(base_input)
        vae_output = self.vae_layer([latent_mean, latent_scale])
        output = self.decoder(vae_output.sample)
        named_output = k_layers.Lambda(lambda x: x, name='output')(output)
        named_kl = k_layers.Lambda(lambda x: x, name='kl')(vae_output.kl)
        return named_output, named_kl



class IWAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            vae_layer,
            nb_of_samples=50,
            base_iwae_loss=None,
            loss_weight=50.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae_layer = vae_layer
        self.nb_of_samples = nb_of_samples
        self.base_iwae_loss = k_losses.MeanSquaredError(reduction='none')
        self.loss_weight = loss_weight

    def call(self, inputs):
        base_input = inputs
        latent_mean, latent_scale = self.encoder(base_input)
        vae_outputs = [
            self.vae_layer([latent_mean, latent_scale])
            for _ in range(self.nb_of_samples)
        ]
        kl = vae_outputs[0].kl
        outputs = [
            self.decoder(vae_output.sample)
            for vae_output in vae_outputs
        ]
        central_output = self.decoder(latent_mean)
        losses_list = [
            tf.reshape(
                self.base_iwae_loss(output, base_input),
                (-1, 1)
            )
            for output in outputs
        ]
        losses = k_layers.Concatenate()(losses_list)
        sample_log_priors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_prior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )
        sample_log_posteriors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_posterior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )

        sample_log_probs = -self.loss_weight * losses + sample_log_priors - sample_log_posteriors
        self.add_metric(tf.reduce_min(losses_list, axis=-1, keepdims=True), name='sample_iwae_loss', aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(outputs[0], base_input), name='mse', aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(central_output, base_input), name='map_mse', aggregation='mean')

        full_loss = -tf.math.reduce_logsumexp(sample_log_probs, axis=-1)
        return full_loss


class GMMIWAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            vae_layer,
            nb_of_samples=20,
            rec_loss_weight=50.0,
            cat_loss_weight=20.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae_layer = vae_layer
        self.nb_of_samples = nb_of_samples
        self.base_iwae_marker_loss = k_losses.MeanSquaredError(reduction='none')
        self.base_iwae_prediction_loss = k_losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.rec_loss_weight = rec_loss_weight
        self.cat_loss_weight = cat_loss_weight

    def call(self, inputs):
        base_input, condition_input = inputs[0], inputs[1]
        latent_mean, latent_scale = self.encoder(inputs)
        vae_outputs = [
            self.vae_layer([latent_mean, latent_scale])
            for _ in range(self.nb_of_samples)
        ]
        outputs = [
            self.decoder(vae_output.sample)
            for vae_output in vae_outputs
        ]
        categorical_outputs = [
            vae_output.sample_mixture_posterior
            for vae_output in vae_outputs
        ]
        central_output = self.decoder(latent_mean)
        marker_losses_list = [
            tf.reshape(
                self.base_iwae_marker_loss(output, base_input),
                (-1, 1)
            )
            for output in outputs
        ]
        categorical_losses_list = [
            tf.reshape(
                self.base_iwae_prediction_loss(condition_input, output),
                (-1, 1)
            )
            for output in categorical_outputs
        ]
        marker_losses = k_layers.Concatenate()(marker_losses_list)
        categorical_losses = k_layers.Concatenate()(categorical_losses_list)
        sample_log_priors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_prior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )
        sample_log_posteriors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_posterior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )

        sample_log_probs = -self.rec_loss_weight * marker_losses - self.cat_loss_weight * categorical_losses
        sample_log_probs = sample_log_probs + sample_log_priors - sample_log_posteriors
        self.add_metric(tf.reduce_min(marker_losses, axis=-1, keepdims=True), name='sample_iwae_loss',
                        aggregation='mean')
        self.add_metric(tf.reduce_min(categorical_losses, axis=-1, keepdims=True), name='sample_cat_loss',
                        aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(outputs[0], base_input), name='mse', aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(central_output, base_input), name='map_mse', aggregation='mean')

        full_loss = -tf.math.reduce_logsumexp(sample_log_probs, axis=-1)
        return full_loss



class TheisEnvGMMIWAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            vae_layer,
            nb_of_samples=20,
            rec_loss_weight=50.0,
            cat_loss_weight=20.0,
            center_index=4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae_layer = vae_layer
        self.nb_of_samples = nb_of_samples
        self.base_iwae_marker_loss = k_losses.MeanSquaredError(reduction='none')
        self.base_iwae_prediction_loss = k_losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.rec_loss_weight = rec_loss_weight
        self.cat_loss_weight = cat_loss_weight
        self.center_index = center_index

    def call(self, inputs):
        base_input, condition_input = inputs[0], inputs[1]
        latent_mean, latent_scale = self.encoder(inputs)
        vae_outputs = [
            self.vae_layer([latent_mean, latent_scale])
            for _ in range(self.nb_of_samples)
        ]
        outputs = [
            self.decoder(vae_output.sample)
            for vae_output in vae_outputs
        ]
        categorical_outputs = [
            vae_output.sample_mixture_posterior
            for vae_output in vae_outputs
        ]
        central_output = self.decoder(latent_mean)
        marker_losses_list = [
            tf.reshape(
                self.base_iwae_marker_loss(output, base_input[:, self.center_index]),
                (-1, 1)
            )
            for output in outputs
        ]
        categorical_losses_list = [
            tf.reshape(
                self.base_iwae_prediction_loss(condition_input, output),
                (-1, 1)
            )
            for output in categorical_outputs
        ]
        marker_losses = k_layers.Concatenate()(marker_losses_list)
        categorical_losses = k_layers.Concatenate()(categorical_losses_list)
        sample_log_priors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_prior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )
        sample_log_posteriors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_posterior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )

        sample_log_probs = -self.rec_loss_weight * marker_losses - self.cat_loss_weight * categorical_losses
        sample_log_probs = sample_log_probs + sample_log_priors - sample_log_posteriors
        self.add_metric(tf.reduce_min(marker_losses, axis=-1, keepdims=True), name='sample_iwae_loss',
                        aggregation='mean')
        self.add_metric(tf.reduce_min(categorical_losses, axis=-1, keepdims=True), name='sample_cat_loss',
                        aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(outputs[0], base_input[:, self.center_index]), name='mse', aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(central_output, base_input[:, self.center_index]), name='map_mse', aggregation='mean')

        full_loss = -tf.math.reduce_logsumexp(sample_log_probs, axis=-1)
        return full_loss


class TheisNeighborhoodGMMIWAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            vae_layer,
            nb_of_samples=20,
            rec_loss_weight=50.0,
            cat_loss_weight=20.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vae_layer = vae_layer
        self.nb_of_samples = nb_of_samples
        self.base_iwae_marker_loss = k_losses.MeanSquaredError(reduction='none')
        self.base_iwae_prediction_loss = k_losses.CategoricalCrossentropy(from_logits=True, reduction='none')
        self.rec_loss_weight = rec_loss_weight
        self.cat_loss_weight = cat_loss_weight

    def call(self, inputs):
        base_input, condition_input = inputs[0], inputs[1]
        latent_mean, latent_scale = self.encoder(inputs)
        vae_outputs = [
            self.vae_layer([latent_mean, latent_scale])
            for _ in range(self.nb_of_samples)
        ]
        outputs = [
            self.decoder(vae_output.sample)
            for vae_output in vae_outputs
        ]
        categorical_outputs = [
            vae_output.sample_mixture_posterior
            for vae_output in vae_outputs
        ]
        central_output = self.decoder(latent_mean)
        marker_losses_list = [
            tf.reshape(
                self.base_iwae_marker_loss(output, base_input[:, 0]),
                (-1, 1)
            )
            for output in outputs
        ]
        categorical_losses_list = [
            tf.reshape(
                self.base_iwae_prediction_loss(condition_input, output),
                (-1, 1)
            )
            for output in categorical_outputs
        ]
        marker_losses = k_layers.Concatenate()(marker_losses_list)
        categorical_losses = k_layers.Concatenate()(categorical_losses_list)
        sample_log_priors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_prior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )
        sample_log_posteriors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_posterior,
                    (-1, 1),
                )
                for vae_output in vae_outputs
            ]
        )

        sample_log_probs = -self.rec_loss_weight * marker_losses - self.cat_loss_weight * categorical_losses
        sample_log_probs = sample_log_probs + sample_log_priors - sample_log_posteriors
        self.add_metric(tf.reduce_min(marker_losses, axis=-1, keepdims=True), name='sample_iwae_loss',
                        aggregation='mean')
        self.add_metric(tf.reduce_min(categorical_losses, axis=-1, keepdims=True), name='sample_cat_loss',
                        aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(central_output, base_input[:, 0]), name='map_mse', aggregation='mean')

        full_loss = -tf.math.reduce_logsumexp(sample_log_probs, axis=-1)
        return full_loss


class MultiModelGMMVAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoders,
            gmm_vae_layer,
    ):
        super(MultiModelGMMVAE, self).__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.gmm_vae_layer = gmm_vae_layer
        self.marker_loss = k_losses.MeanSquaredError(reduction='none')
        self.marker_loss_weight = 5.0

    def call(self, inputs):
        base_input, condition_input = inputs[0], inputs[1]
        latent_mean, latent_scale = self.encoder(base_input)
        gmm_vae_output = self.gmm_vae_layer([latent_mean, latent_scale])
        component_means = k_layers.Input(tensor=self.gmm_vae_layer.component_means)
        decoder_inputs = [
            gmm_vae_output.sample - component_means[decoder_index]
            for decoder_index, decoder in enumerate(self.decoders)
        ]
        outputs = [
            decoder(decoder_input)
            for decoder_input, decoder in zip(decoder_inputs, self.decoders)
        ]
        losses = [
            tf.reshape(
                self.marker_loss(output, base_input),
                (-1, 1)
            )
            for output in outputs
        ]
        losses = k_layers.Concatenate()(losses)

        pre_log_sum_exp = -losses * self.marker_loss_weight + gmm_vae_output.sample_mixture_posterior
        self.add_metric(tf.reduce_min(losses, axis=-1, keepdims=True), name='sample_iwae_loss', aggregation='mean')

        return -tf.reduce_logsumexp(pre_log_sum_exp, axis=-1)


class MultiModelConditionalGMMIWAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoders,
            gmm_vae_layer,
            general_decoder,
    ):
        super(MultiModelConditionalGMMIWAE, self).__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.general_decoder = general_decoder
        self.gmm_vae_layer = gmm_vae_layer
        self.nb_of_samples = 20
        self.marker_loss = k_losses.Huber(delta=1.0, reduction='none')
        self.marker_loss_weight = 50.0

    def call(self, inputs):
        base_input, condition_input = inputs[0], inputs[1]
        latent_mean, latent_scale = self.encoder(base_input)
        gmm_vae_samples = [
                self.gmm_vae_layer([latent_mean, latent_scale, condition_input])
                for _ in range(self.nb_of_samples)
        ]
        component_means = k_layers.Input(tensor=self.gmm_vae_layer.component_means)
        decoder_outputs = [
            sum(
                [
                    (
                        decoder(gmm_vae_output.sample - component_means[decoder_index]) + self.general_decoder(gmm_vae_output.sample)
                    )
                        *\
                    tf.reshape(condition_input[:, decoder_index], (-1, 1))
                    for decoder_index, decoder in enumerate(self.decoders)
                ]
            )
            for gmm_vae_output in gmm_vae_samples
        ]
        central_output = sum(
            [
                (
                        decoder(latent_mean - component_means[decoder_index]) + self.general_decoder(latent_mean)
                )
                * \
                tf.reshape(condition_input[:, decoder_index], (-1, 1))
                for decoder_index, decoder in enumerate(self.decoders)
            ]
        )
        marker_losses_list = [
            tf.reshape(
                self.marker_loss(output, base_input),
                (-1, 1)
            )
            for output in decoder_outputs
        ]
        marker_losses = k_layers.Concatenate()(marker_losses_list)
        sample_log_priors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_prior,
                    (-1, 1),
                )
                for vae_output in gmm_vae_samples
            ]
        )
        sample_log_posteriors = k_layers.Concatenate()(
            [
                tf.reshape(
                    vae_output.sample_log_posterior,
                    (-1, 1),
                )
                for vae_output in gmm_vae_samples
            ]
        )
        vae_pairwise_components_distances_matrix = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=0.1).matrix(
            x1=self.gmm_vae_layer.component_means, x2=self.gmm_vae_layer.component_means,
        )
        vae_pairwise_components_distances = tf.reduce_sum(vae_pairwise_components_distances_matrix)
        sample_log_probs = -self.marker_loss_weight * marker_losses
        sample_log_probs = sample_log_probs + sample_log_priors - sample_log_posteriors
        self.add_metric(tf.reduce_min(marker_losses, axis=-1, keepdims=True), name='sample_iwae_loss',
                        aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(decoder_outputs[0], base_input), name='mse', aggregation='mean')
        self.add_metric(k_losses.MeanSquaredError()(central_output, base_input), name='map_mse', aggregation='mean')
        self.add_metric(vae_pairwise_components_distances, name='pd', aggregation='mean')
        full_loss = -tf.math.reduce_logsumexp(sample_log_probs, axis=-1) + vae_pairwise_components_distances
        return full_loss, central_output


class WinterCellEnvironmentAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            marker_loss=None,
            entropy_weight=1e-5,
    ):
        super(WinterCellEnvironmentAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size,
            output_dim=self.encoder.latent_size,
        )
        self.marker_loss = marker_loss if marker_loss is not None else k_losses.MeanSquaredError()
        self.entropy_weight = entropy_weight

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking = self.ranking_layer(full_output)
        ranking = k_layers.Flatten()(ranking)

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
                           tf.reshape(
                               tf.range(0, self.max_neighborhood_size),
                               (1, -1),
                           )
            )
        )
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        encoding = k_layers.GlobalAvgPool1D()(full_output)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size)(encoding)
        position_encodings_dim_permuted = k_layers.Permute([2, 1])(position_encodings)
        position_encodings_dim_permuted_permuted = tf.matmul(position_encodings_dim_permuted, sort_matrix)
        position_encodings_permuted = k_layers.Permute([2, 1])(position_encodings_dim_permuted_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        permutation_matrix = inputs[-1]
        is_sensible = tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
        )
        decoded = self.decoder([decoder_input, inputs[-1]])
        losses = tf.reduce_sum([
            self.marker_loss(inputs[1][:, i, :], decoded[:, i, :]) * is_sensible[:, i]
            for i in range(self.max_neighborhood_size)
        ], axis=0)
        soft_sort_entropies = tf.reshape(
            tfd.Categorical(
                probs=tf.reshape(
                    tf.maximum(sort_matrix, 1e-5),
                    shape=(-1, self.max_neighborhood_size),
                )).entropy(),
            shape=(-1, self.max_neighborhood_size),
        )
        soft_sort_entropies_sums = tf.reduce_mean(soft_sort_entropies, axis=-1)
        self.add_metric(soft_sort_entropies_sums, name='ent', aggregation='mean')
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0)
        return full_loss + self.entropy_weight * soft_sort_entropies_sums


class WinterCellEnvironmentAEV2(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            marker_loss=None,
            entropy_weight=1e-5,
    ):
        super(WinterCellEnvironmentAEV2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size,
            output_dim=self.encoder.latent_size,
        )
        self.marker_loss = marker_loss if marker_loss is not None else k_losses.MeanSquaredError()
        self.entropy_weight = entropy_weight

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking = self.ranking_layer(full_output)
        ranking = k_layers.Flatten()(ranking)

        permutation_matrix = inputs[-1]
        is_sensible = k_layers.Input(tensor=tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
        ))
        is_sensible_extended = k_layers.Reshape((self.max_neighborhood_size, 1))(is_sensible)
        normalizing_factor = k_layers.GlobalAveragePooling1D()(is_sensible_extended) * self.max_neighborhood_size
        normalizing_factor = k_layers.Maximum()([normalizing_factor, k_backend.ones_like(normalizing_factor)])

        full_output = k_layers.Multiply()([full_output, is_sensible_extended])
        encoding = k_layers.Add()([
            full_output[:, i, :]
            for i in range(self.max_neighborhood_size)
        ]) / normalizing_factor

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
                           tf.reshape(
                               tf.range(0, self.max_neighborhood_size),
                               (1, -1),
                           )
            )
        )
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size)(encoding)
        position_encodings_dim_permuted = k_layers.Permute([2, 1])(position_encodings)
        position_encodings_dim_permuted_permuted = tf.matmul(position_encodings_dim_permuted, sort_matrix)
        position_encodings_permuted = k_layers.Permute([2, 1])(position_encodings_dim_permuted_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        decoded = self.decoder([decoder_input, inputs[-1]])
        losses = tf.reduce_sum([
            self.marker_loss(inputs[1][:, i, :], decoded[:, i, :]) * is_sensible[:, i]
            for i in range(self.max_neighborhood_size)
        ], axis=0)
        soft_sort_entropies = tf.reshape(
            tfd.Categorical(
                probs=tf.reshape(
                    tf.maximum(sort_matrix, 1e-5),
                    shape=(-1, self.max_neighborhood_size),
                )).entropy(),
            shape=(-1, self.max_neighborhood_size),
        )
        soft_sort_entropies_sums = tf.reduce_mean(soft_sort_entropies, axis=-1)
        self.add_metric(soft_sort_entropies_sums, name='ent', aggregation='mean')
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0)
        return full_loss# + self.entropy_weight * soft_sort_entropies_sums


class WinterCellEnvironmentGraphSetAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            marker_loss=None,
            entropy_weight=1e-5,
    ):
        super(WinterCellEnvironmentGraphSetAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size,
            output_dim=self.encoder.latent_size,
        )
        self.marker_loss = marker_loss if marker_loss is not None else k_losses.MeanSquaredError()
        self.entropy_weight = entropy_weight

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking = self.ranking_layer(full_output)
        ranking = k_layers.Flatten()(ranking)

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
                           tf.reshape(
                               tf.range(0, self.max_neighborhood_size),
                               (1, -1),
                           )
            )
        )
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        encoding = k_layers.GlobalAvgPool1D()(full_output)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size)(encoding)
        position_encodings_dim_permuted = k_layers.Permute([2, 1])(position_encodings)
        position_encodings_dim_permuted_permuted = tf.matmul(position_encodings_dim_permuted, sort_matrix)
        position_encodings_permuted = k_layers.Permute([2, 1])(position_encodings_dim_permuted_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        permutation_matrix = inputs[-1]
        is_sensible = tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
        )
        decoded = self.decoder([decoder_input, inputs[-2], inputs[-1]])
        losses = tf.reduce_sum([
            self.marker_loss(inputs[0][:, i, :], decoded[:, i, :]) * is_sensible[:, i]
            for i in range(self.max_neighborhood_size)
        ], axis=0)
        soft_sort_entropies = tf.reshape(
            tfd.Categorical(
                probs=tf.reshape(
                    tf.maximum(sort_matrix, 1e-5),
                    shape=(-1, self.max_neighborhood_size),
                )).entropy(),
            shape=(-1, self.max_neighborhood_size),
        )
        soft_sort_entropies_sums = tf.reduce_mean(soft_sort_entropies, axis=-1)
        self.add_metric(soft_sort_entropies_sums, name='ent', aggregation='mean')
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0)
        return full_loss + self.entropy_weight * soft_sort_entropies_sums
    
    
class WinterCellEnvironmentGraphSetAEV2(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            marker_loss=None,
            entropy_weight=1e-5,
    ):
        super(WinterCellEnvironmentGraphSetAEV2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size,
            output_dim=self.encoder.latent_size,
        )
        self.marker_loss = marker_loss if marker_loss is not None else k_losses.MeanSquaredError()
        self.entropy_weight = entropy_weight

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking = self.ranking_layer(full_output)
        ranking = k_layers.Flatten()(ranking)

        permutation_matrix = inputs[-1]
        is_sensible = k_layers.Input(tensor=tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
        ))
        is_sensible_extended = k_layers.Reshape((self.max_neighborhood_size, 1))(is_sensible)
        normalizing_factor = k_layers.GlobalAveragePooling1D()(is_sensible_extended) * self.max_neighborhood_size
        normalizing_factor = k_layers.Maximum()([normalizing_factor, k_backend.ones_like(normalizing_factor)])

        full_output = k_layers.Multiply()([full_output, is_sensible_extended])
        encoding = k_layers.Add()([
            full_output[:, i, :]
            for i in range(self.max_neighborhood_size)
        ]) / normalizing_factor

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
                           tf.reshape(
                               tf.range(0, self.max_neighborhood_size),
                               (1, -1),
                           )
            )
        )
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size)(encoding)
        position_encodings_dim_permuted = k_layers.Permute([2, 1])(position_encodings)
        position_encodings_dim_permuted_permuted = tf.matmul(position_encodings_dim_permuted, sort_matrix)
        position_encodings_permuted = k_layers.Permute([2, 1])(position_encodings_dim_permuted_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        permutation_matrix = inputs[-1]
        decoded = self.decoder([decoder_input, inputs[-2], inputs[-1]])
        losses = tf.reduce_sum([
            self.marker_loss(inputs[0][:, i, :], decoded[:, i, :]) * is_sensible[:, i]
            for i in range(self.max_neighborhood_size)
        ], axis=0)
        soft_sort_entropies = tf.reshape(
            tfd.Categorical(
                probs=tf.reshape(
                    tf.maximum(sort_matrix, 1e-5),
                    shape=(-1, self.max_neighborhood_size),
                )).entropy(),
            shape=(-1, self.max_neighborhood_size),
        )
        soft_sort_entropies_sums = tf.reduce_mean(soft_sort_entropies, axis=-1)
        self.add_metric(soft_sort_entropies_sums, name='ent', aggregation='mean')
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0)
        return full_loss + self.entropy_weight * soft_sort_entropies_sums

    
class WinterCellGraphAE(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            final_loss=None,
    ):
        super(WinterCellGraphAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size ** 2,
            output_dim=self.encoder.latent_size,
        )
        self.final_loss = final_loss if final_loss is not None else k_losses.MeanSquaredError()

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking_output = full_output[:, ::self.max_neighborhood_size]
        ranking = self.ranking_layer(ranking_output)
        ranking = k_layers.Flatten()(ranking)

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
            tf.reshape(
                tf.range(0, self.max_neighborhood_size ** 2),
                (1, -1),
            )
            )
        )
        squared_position_encodings = k_layers.Reshape(
            (self.max_neighborhood_size, self.max_neighborhood_size, self.encoder.latent_size)
        )(position_encodings)
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        sort_matrix = tf.expand_dims(sort_matrix, axis=1)
        encoding = k_layers.GlobalAvgPool1D()(full_output)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size ** 2)(encoding)
        position_encodings_dim_permuted_left = k_layers.Permute([1, 3, 2])(squared_position_encodings)
        position_encodings_dim_permuted_permuted_left = tf.matmul(position_encodings_dim_permuted_left, sort_matrix)
        position_encodings_permuted_left = k_layers.Permute([1, 3, 2])(position_encodings_dim_permuted_permuted_left)
        position_encodings_dim_permuted_right = k_layers.Permute([3, 2, 1])(position_encodings_permuted_left)
        position_encodings_dim_permuted_permuted_right = tf.matmul(position_encodings_dim_permuted_right, sort_matrix)
        position_encodings_permuted = k_layers.Permute([3, 2, 1])(position_encodings_dim_permuted_permuted_right)
        position_encodings_permuted = k_layers.Reshape(
            (self.max_neighborhood_size ** 2, self.encoder.latent_size)
        )(position_encodings_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        permutation_matrix = inputs[-1]
        is_sensible = tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
            )
        decoded = self.decoder([decoder_input, inputs[-1]])
        losses = sum([
            self.final_loss(
                decoded[:, i, :], inputs[0][:, i, :]
            ) * is_sensible[:, i // self.max_neighborhood_size] * is_sensible[:, i % self.max_neighborhood_size]
            for i in range(self.max_neighborhood_size ** 2)
        ])
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0) ** 2
        return full_loss


class WinterCellGraphAEWithEnt(k_models.Model):

    def __init__(
            self,
            encoder,
            decoder,
            max_neigborhood_size=10,
            temperature=0.5,
            final_loss=None,
            entropy_reg_coef=1e-5,
    ):
        super(WinterCellGraphAEWithEnt, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ranking_layer = k_layers.Dense(1)
        self.temperature = temperature
        self.max_neighborhood_size = max_neigborhood_size
        self.position_embeddings = k_layers.Embedding(
            input_dim=self.max_neighborhood_size ** 2,
            output_dim=self.encoder.latent_size,
        )
        self.final_loss = final_loss if final_loss is not None else k_losses.MeanSquaredError()
        self.entropy_reg_coef = entropy_reg_coef

    def call(self, inputs):
        full_output = self.encoder(inputs)
        ranking_output = full_output[:, ::self.max_neighborhood_size]
        ranking = self.ranking_layer(ranking_output)
        ranking = k_layers.Flatten()(ranking)

        position_encodings = self.position_embeddings(
            k_layers.Input(tensor=
            tf.reshape(
                tf.range(0, self.max_neighborhood_size ** 2),
                (1, -1),
            )
            )
        )
        squared_position_encodings = k_layers.Reshape(
            (self.max_neighborhood_size, self.max_neighborhood_size, self.encoder.latent_size)
        )(position_encodings)
        sort_matrix = tfp.math.soft_sorting_matrix(ranking, self.temperature)
        sort_matrix = tf.expand_dims(sort_matrix, axis=1)
        encoding = k_layers.GlobalAvgPool1D()(full_output)
        encoded_repeated = k_layers.RepeatVector(self.max_neighborhood_size ** 2)(encoding)
        position_encodings_dim_permuted_left = k_layers.Permute([1, 3, 2])(squared_position_encodings)
        position_encodings_dim_permuted_permuted_left = tf.matmul(position_encodings_dim_permuted_left, sort_matrix)
        position_encodings_permuted_left = k_layers.Permute([1, 3, 2])(position_encodings_dim_permuted_permuted_left)
        position_encodings_dim_permuted_right = k_layers.Permute([3, 2, 1])(position_encodings_permuted_left)
        position_encodings_dim_permuted_permuted_right = tf.matmul(position_encodings_dim_permuted_right, sort_matrix)
        position_encodings_permuted = k_layers.Permute([3, 2, 1])(position_encodings_dim_permuted_permuted_right)
        position_encodings_permuted = k_layers.Reshape(
            (self.max_neighborhood_size ** 2, self.encoder.latent_size)
        )(position_encodings_permuted)
        decoder_input = k_layers.Add()([position_encodings_permuted, encoded_repeated])
        permutation_matrix = inputs[-1]
        is_sensible = tf.cast(
            tf.math.reduce_sum(permutation_matrix, axis=-1) > 1.0,
            'float32',
            )
        decoded = self.decoder([decoder_input, inputs[-1]])
        losses = tf.reduce_sum([
            self.final_loss(
                decoded[:, i, :], inputs[0][:, i, :]
            ) * is_sensible[:, i // self.max_neighborhood_size] * is_sensible[:, i % self.max_neighborhood_size]
            for i in range(self.max_neighborhood_size ** 2)
        ], axis=0)
        soft_sort_entropies = tf.reshape(
            tfd.Categorical(
                probs=tf.reshape(
                    tf.maximum(sort_matrix, 1e-5),
                    shape=(-1, self.max_neighborhood_size),
                )).entropy(),
            shape=(-1, self.max_neighborhood_size),
        )
        soft_sort_entropies_sums = tf.reduce_mean(soft_sort_entropies, axis=-1)
        self.add_metric(soft_sort_entropies_sums, name='ent', aggregation='mean')
        full_loss = losses / tf.maximum(tf.math.reduce_sum(is_sensible, axis=-1), 1.0) ** 2
        return full_loss + self.entropy_reg_coef * soft_sort_entropies_sums


class BaseWinterCellEnvironmentAE(WinterCellEnvironmentAE):

    def __init__(
            self,
            output_size,
            max_neighborhood_size=10,
            marker_loss=None,
    ):
        super(BaseWinterCellEnvironmentAE, self).__init__(
            encoder=encoders.FullOutputBaseGNNEncoder(),
            decoder=decoders.FullOutputGNNDecoder(output_size=output_size),
            max_neigborhood_size=max_neighborhood_size,
            marker_loss=marker_loss,
        )

        
class BaseWinterCellEnvironmentAEV2(WinterCellEnvironmentAEV2):

    def __init__(
            self,
            output_size,
            max_neighborhood_size=10,
            latent_size=64,
            marker_loss=None,
            layer_size=128,
            intermediate_layer_size=256,
    ):
        super(BaseWinterCellEnvironmentAEV2, self).__init__(
            encoder=encoders.FullOutputBaseGNNEncoder(latent_size=latent_size, intermediate_layer_size=intermediate_layer_size, layer_size=layer_size),
            decoder=decoders.FullOutputGNNDecoder(output_size=output_size, intermediate_layer_size=intermediate_layer_size, layer_size=layer_size),
            max_neigborhood_size=max_neighborhood_size,
            marker_loss=marker_loss,
        )
        self.output_size = output_size
        self.max_neighborhood_size = max_neighborhood_size
        self.latent_size = latent_size
        self.layer_size = layer_size
        self.intermediate_layer_size = intermediate_layer_size


class BaseWinterCellEnvironmentGraphSetAE(WinterCellEnvironmentGraphSetAE):

    def __init__(
            self,
            output_size,
            max_neighborhood_size=10,
            marker_loss=None,
    ):
        print('ok')
        super(BaseWinterCellEnvironmentGraphSetAE, self).__init__(
            encoder=encoders.FullOutputBaseGNNSetAndGraphEncoder(
                ),
            decoder=decoders.FullOutputGNNSetGraphDecoder(
                output_size=output_size,
                ),
            max_neigborhood_size=max_neighborhood_size,
            marker_loss=marker_loss,
        )
        
        
class BaseWinterCellEnvironmentGraphSetAEV2(WinterCellEnvironmentGraphSetAEV2):

    def __init__(
            self,
            output_sizeALL_CELL_DATA_COLS_NB,
            max_neighborhood_size=10,
            marker_loss=None,
    ):
        super(BaseWinterCellEnvironmentGraphSetAEV2, self).__init__(
            encoder=encoders.FullOutputBaseGNNSetAndGraphEncoder(
                ),
            decoder=decoders.FullOutputGNNSetGraphDecoder(
                output_size=output_size,
                ),
            max_neigborhood_size=max_neighborhood_size,
            marker_loss=marker_loss,
        )


class BaseWinterCellGraphAE(WinterCellGraphAE):

    def __init__(self, output_size, max_neighborhood_size=10):
        super(BaseWinterCellGraphAE, self).__init__(
            encoder=encoders.FullOutputBaseGNNEncoder(),
            decoder=decoders.FullOutputGNNDecoder(output_size=output_size),
            max_neigborhood_size=max_neighborhood_size,
        )


class BaseWinterCellGraphAEWithEnt(WinterCellGraphAEWithEnt):

    def __init__(self, output_size, max_neighborhood_size=10):
        super(BaseWinterCellGraphAEWithEnt, self).__init__(
            encoder=encoders.FullOutputBaseGNNEncoder(),
            decoder=decoders.FullOutputGNNDecoder(output_size=output_size),
            max_neigborhood_size=max_neighborhood_size,
        )


class BaseMarkerVAE(VAE):

    def __init__(self):
        super(BaseMarkerVAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAELayer(),
        )


class BaseCellVAE(VAE):

    def __init__(self):
        super(BaseCellVAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAELayer(),
        )


class BaseMarkerIWAE(IWAE):

    def __init__(self):
        super(BaseMarkerIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAELayer(),
        )



class BaseCellIWAE(IWAE):

    def __init__(self):
        super(BaseCellIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAELayer(),
        )


class BaseMarkerGMMIWAE(GMMIWAE):

    def __init__(self):
        super(BaseMarkerGMMIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAEGMMLayer(),
        )


class BaseMarkerGNNGMMIWAE(TheisEnvGMMIWAE):

    def __init__(self):
        super(BaseMarkerGNNGMMIWAE, self).__init__(
            encoder=encoders.BaseGNNVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAEGMMLayer(),
        )


class BaseCellGNNGMMIWAE(TheisNeighborhoodGMMIWAE):

    def __init__(self):
        super(BaseCellGNNGMMIWAE, self).__init__(
            encoder=encoders.BaseGNNVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAEGMMLayer(components_scale=1.0, starting_components_scale=3.0),
        )


class BaseCellGNNHierarchicalGMMIWAE(TheisNeighborhoodGMMIWAE):

    def __init__(self):
        super(BaseCellGNNHierarchicalGMMIWAE, self).__init__(
            encoder=encoders.BaseGNNVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.HierarchicalVAEGMMLayer(
                components_scale=1.0,
                starting_components_scale=3.0,
            ),
            rec_loss_weight=50.0,
            cat_loss_weight=0.0,
        )


class BaseCellGMMIWAE(GMMIWAE):

    def __init__(self):
        super(BaseCellGMMIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoder=decoders.BaseDecoder(),
            vae_layer=vae_layer.VAEGMMLayer(),
        )


class TLDCellGMMIWAE(GMMIWAE):

    def __init__(self):
        super(TLDCellGMMIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(latent_size=32),
            decoder=decoders.ThreeLayerDecoder(),
            vae_layer=vae_layer.VAEGMMLayer(nb_components=15, latent_size=32),
        )


class BaseMarkerMultiModelGMMVAE(MultiModelGMMVAE):

    def __init__(self, nb_of_components: int = 32):
        super(BaseMarkerMultiModelGMMVAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoders=[decoders.BaseDecoder() for _ in range(nb_of_components)],
            gmm_vae_layer=vae_layer.VAEGMMLayer(nb_components=nb_of_components),
        )


class BaseCellMultiModelGMMVAE(MultiModelGMMVAE):

    def __init__(self, nb_of_components: int = 32):
        super(BaseCellMultiModelGMMVAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(),
            decoders=[
                decoders.BaseDecoder()
                for _ in range(nb_of_components)
            ],
            gmm_vae_layer=vae_layer.VAEGMMLayer(nb_components=nb_of_components),
        )


class BaseCellMultiModelConditionalGMMIWAE(MultiModelConditionalGMMIWAE):

    def __init__(self, output_size=40):
        super(BaseCellMultiModelConditionalGMMIWAE, self).__init__(
            encoder=encoders.BaseVAEEncoder(
                output_size=output_size,
                first_layer_size=1024,
                second_layer_size=512,
                latent_size=64,
            ),
            decoders=[
                decoders.BaseDecoder(
                    output_size=output_size,
                    l1_weight=0.0001,
                )
                for _ in range(DEFAULT_NB_OF_CELL_TYPES)
            ],
            general_decoder=decoders.BaseDecoder(
                output_size=im_misc.ALL_CELL_DATA_COLS_NB,
                l1_weight=0.0001,
            ),
            gmm_vae_layer=vae_layer.VAEGMMLayerWithCondition(
                components_scale=0.7,
            ),
        )

class BaseGNNMarkerAutoEncoder(AutoEncoder):

    def __init__(self):
        super(BaseGNNMarkerAutoEncoder, self).__init__(
            encoder=encoders.BaseGNNEncoder(),
            decoder=decodersBaseDecoder(),
        )


class BaseGNNCellAutoEncoder(AutoEncoder):

    def __init__(self):
        super(BaseGNNCellAutoEncoder, self).__init__(
            encoder=encoders.BaseGNNEncoder(),
            decoder=decoders.BaseDecoder(output_size=im_misc.ALL_CELL_DATA_COLS_NB),
        )
