from keras import layers as k_layers
from keras import models as k_models
from keras import regularizers as k_regularizers
import tensorflow as tf


DEFAULT_NB_OF_MARKERS = 40


class BaseDecoder(k_models.Model):

    def __init__(
            self,
            output_size=DEFAULT_NB_OF_MARKERS,
            l2_weight=0.0,
            l1_weight=0.0,
    ):
        super(BaseDecoder, self).__init__()
        self.output_size = output_size
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.decoder_layer = k_layers.Dense(
            units=self.output_size,
            kernel_regularizer=k_regularizers.L1L2(
                l1=self.l1_weight,
                l2=self.l2_weight,
            )
        )

    def call(self, inputs):
        return self.decoder_layer(inputs)


class ThreeLayerDecoder(k_models.Model):

    def __init__(
            self,
            output_size=DEFAULT_NB_OF_MARKERS,
            l2_weight=0.0,
            l1_weight=0.0,
            first_layer_size=64,
            second_layer_size=64,
            activation='tanh',
    ):
        super(ThreeLayerDecoder, self).__init__()
        self.output_size = output_size
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.activation = activation
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.decoder_first_layer = k_layers.Dense(
            units=self.first_layer_size,
            kernel_regularizer=k_regularizers.L1L2(
                l1=self.l1_weight,
                l2=self.l2_weight,
            ),
            activation=self.activation,
        )
        self.decoder_second_layer = k_layers.Dense(
            units=self.second_layer_size,
            kernel_regularizer=k_regularizers.L1L2(
                l1=self.l1_weight,
                l2=self.l2_weight,
            ),
            activation=self.activation,
        )
        self.decoder_final_layer = k_layers.Dense(
            units=self.output_size,
            kernel_regularizer=k_regularizers.L1L2(
                l1=self.l1_weight,
                l2=self.l2_weight,
            ),
        )

    def call(self, inputs):
        aux = self.decoder_first_layer(inputs)
        aux = self.decoder_second_layer(aux)
        return self.decoder_final_layer(aux)


class FullOutputGNNDecoder(k_models.Model):
        def __init__(
                self,
                layer_size=128,
                intermediate_layer_size=256,
                num_heads=4,
                key_size=64,
                key_dim=32,
                output_size=DEFAULT_NB_OF_MARKERS,
        ):
            super(FullOutputGNNDecoder, self).__init__()
            self.encoding_layer = k_layers.Dense(layer_size)

            self.first_encoder_layer_k = k_layers.Dense(key_size)
            self.second_encoder_layer_k = k_layers.Dense(key_size)
            self.output_layer_k = k_layers.Dense(key_size)

            self.first_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
            self.second_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
            self.output_layer_v = k_layers.Dense(output_size)

            self.first_encoder_layer_q = k_layers.Dense(key_size)
            self.second_encoder_layer_q = k_layers.Dense(key_size)
            self.output_layer_q = k_layers.Dense(key_size)

            self.first_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')
            self.second_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')

            self.first_encoder_layer_ff_2 = k_layers.Dense(layer_size)
            self.second_encoder_layer_ff_2 = k_layers.Dense(layer_size)

            self.first_encoder_att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
            )
            self.second_encoder_att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
            )
            self.output_encoder_att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=output_size,
            )
            self.first_layer_norm = k_layers.LayerNormalization()
            self.second_layer_norm = k_layers.LayerNormalization()

        def call(self, inputs):
            marker_input, mask_adjacency = inputs[0], inputs[1]
            first_layer_input = self.encoding_layer(marker_input)
            first_encoder_k = self.first_encoder_layer_k(first_layer_input)
            first_encoder_q = self.first_encoder_layer_q(first_layer_input)
            first_encoder_v = self.first_encoder_layer_v(first_layer_input)
            first_encoder = self.first_encoder_att(
                query=first_encoder_v,
                value=first_encoder_q,
                key=first_encoder_k,
                attention_mask=mask_adjacency,
            )
            first_encoder = self.first_encoder_layer_ff_1(first_encoder)
            first_encoder = self.first_encoder_layer_ff_2(first_encoder)
            first_encoder = k_layers.Add()([first_layer_input, first_encoder])
            first_encoder = self.first_layer_norm(first_encoder)

            second_encoder_k = self.second_encoder_layer_k(first_encoder)
            second_encoder_q = self.second_encoder_layer_q(first_encoder)
            second_encoder_v = self.second_encoder_layer_v(first_encoder)
            second_encoder = self.second_encoder_att(
                query=second_encoder_v,
                value=second_encoder_q,
                key=second_encoder_k,
                attention_mask=mask_adjacency,
            )
            second_encoder = self.second_encoder_layer_ff_1(second_encoder)
            second_encoder = self.second_encoder_layer_ff_2(second_encoder)
            second_encoder = k_layers.Add()([first_encoder, second_encoder])
            second_encoder = self.second_layer_norm(second_encoder)

            output_encoder_k = self.output_layer_k(second_encoder)
            output_encoder_q = self.output_layer_q(second_encoder)
            output_encoder_v = self.output_layer_v(second_encoder)

            output = self.output_encoder_att(
                query=output_encoder_v,
                value=output_encoder_q,
                key=output_encoder_k,
                attention_mask=mask_adjacency,
            )
            return output

class FullOutputGNNSetGraphDecoder(k_models.Model):
    def __init__(
            self,
            layer_size=128,
            intermediate_layer_size=256,
            num_heads=4,
            key_size=64,
            key_dim=32,
            output_size=DEFAULT_NB_OF_MARKERS,
    ):
        super(FullOutputGNNSetGraphDecoder, self).__init__()

        self.set_encoding_layer = k_layers.Dense(layer_size)
        self.graph_encoding_layer = k_layers.Dense(layer_size)

        self.first_set_encoder_layer_k = k_layers.Dense(key_size)
        self.first_graph_encoder_layer_k = k_layers.Dense(key_size)
        self.second_set_encoder_layer_k = k_layers.Dense(key_size)
        self.second_graph_encoder_layer_k = k_layers.Dense(key_size)
        self.latent_set_layer_k = k_layers.Dense(key_size)
        self.latent_graph_layer_k = k_layers.Dense(key_size)

        self.first_set_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.first_graph_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.second_set_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.second_graph_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.latent_set_layer_v = k_layers.Dense(output_size)
        self.latent_graph_layer_v = k_layers.Dense(output_size)

        self.first_set_encoder_layer_q = k_layers.Dense(key_size)
        self.first_graph_encoder_layer_q = k_layers.Dense(key_size)
        self.second_set_encoder_layer_q = k_layers.Dense(key_size)
        self.second_graph_encoder_layer_q = k_layers.Dense(key_size)
        self.latent_set_layer_q = k_layers.Dense(key_size)
        self.latent_graph_layer_q = k_layers.Dense(key_size)

        self.first_set_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')
        self.first_graph_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')
        self.second_set_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')
        self.second_graph_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')

        self.first_set_encoder_layer_ff_2 = k_layers.Dense(layer_size)
        self.first_graph_encoder_layer_ff_2 = k_layers.Dense(layer_size)
        self.second_set_encoder_layer_ff_2 = k_layers.Dense(layer_size)
        self.second_graph_encoder_layer_ff_2 = k_layers.Dense(layer_size)

        self.first_set_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.first_graph_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.second_set_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.second_graph_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.latent_set_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.latent_graph_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.first_set_layer_norm = k_layers.LayerNormalization()
        self.first_graph_layer_norm = k_layers.LayerNormalization()
        self.second_set_layer_norm = k_layers.LayerNormalization()
        self.second_graph_layer_norm = k_layers.LayerNormalization()

    def call(self, inputs):
        marker_input, set_mask, graph_mask = inputs[0], inputs[1], inputs[2]
        first_set_layer_input = self.set_encoding_layer(marker_input)
        first_graph_layer_input = self.graph_encoding_layer(marker_input)

        first_set_encoder_k = self.first_set_encoder_layer_k(first_set_layer_input)
        first_set_encoder_q = self.first_set_encoder_layer_q(first_set_layer_input)
        first_set_encoder_v = self.first_set_encoder_layer_v(first_set_layer_input)
        first_set_encoder = self.first_set_encoder_att(
            query=first_set_encoder_v,
            value=first_set_encoder_q,
            key=first_set_encoder_k,
            attention_mask=set_mask,
        )
        first_set_encoder = self.first_set_encoder_layer_ff_1(first_set_encoder)
        first_set_encoder = self.first_set_encoder_layer_ff_2(first_set_encoder)
        first_set_encoder = k_layers.Add()([first_set_layer_input, first_set_encoder])
        first_set_encoder = self.first_set_layer_norm(first_set_encoder)

        first_graph_encoder_k = self.first_graph_encoder_layer_k(first_graph_layer_input)
        first_graph_encoder_q = self.first_graph_encoder_layer_q(first_graph_layer_input)
        first_graph_encoder_v = self.first_graph_encoder_layer_v(first_graph_layer_input)
        first_graph_encoder = self.first_graph_encoder_att(
            query=first_graph_encoder_v,
            value=first_graph_encoder_q,
            key=first_graph_encoder_k,
            attention_mask=graph_mask,
        )
        first_graph_encoder = self.first_graph_encoder_layer_ff_1(first_graph_encoder)
        first_graph_encoder = self.first_graph_encoder_layer_ff_2(first_graph_encoder)
        first_graph_encoder = k_layers.Add()([first_graph_layer_input, first_graph_encoder])
        first_graph_encoder = self.first_graph_layer_norm(first_graph_encoder)

        first_encoder = k_layers.Add()([first_graph_encoder, first_set_encoder])

        second_set_encoder_k = self.second_set_encoder_layer_k(first_encoder)
        second_set_encoder_q = self.second_set_encoder_layer_q(first_encoder)
        second_set_encoder_v = self.second_set_encoder_layer_v(first_encoder)
        second_set_encoder = self.second_set_encoder_att(
            query=second_set_encoder_v,
            value=second_set_encoder_q,
            key=second_set_encoder_k,
            attention_mask=set_mask,
        )
        second_set_encoder = self.second_set_encoder_layer_ff_1(second_set_encoder)
        second_set_encoder = self.second_set_encoder_layer_ff_2(second_set_encoder)
        second_set_encoder = k_layers.Add()([first_encoder, second_set_encoder])
        second_set_encoder = self.second_set_layer_norm(second_set_encoder)

        second_graph_encoder_k = self.second_graph_encoder_layer_k(first_encoder)
        second_graph_encoder_q = self.second_graph_encoder_layer_q(first_encoder)
        second_graph_encoder_v = self.second_graph_encoder_layer_v(first_encoder)
        second_graph_encoder = self.second_graph_encoder_att(
            query=second_graph_encoder_v,
            value=second_graph_encoder_q,
            key=second_graph_encoder_k,
            attention_mask=graph_mask,
        )
        second_graph_encoder = self.second_graph_encoder_layer_ff_1(second_graph_encoder)
        second_graph_encoder = self.second_graph_encoder_layer_ff_2(second_graph_encoder)
        second_graph_encoder = k_layers.Add()([first_encoder, second_graph_encoder])
        second_graph_encoder = self.second_graph_layer_norm(second_graph_encoder)

        second_encoder = k_layers.Add()([second_graph_encoder, second_set_encoder])

        latent_set_encoder_k = self.latent_set_layer_k(second_encoder)
        latent_set_encoder_q = self.latent_set_layer_q(second_encoder)
        latent_set_encoder_v = self.latent_set_layer_v(second_encoder)

        latent_set = self.latent_set_encoder_att(
            query=latent_set_encoder_v,
            value=latent_set_encoder_q,
            key=latent_set_encoder_k,
            attention_mask=set_mask,
        )

        latent_graph_encoder_k = self.latent_graph_layer_k(second_encoder)
        latent_graph_encoder_q = self.latent_graph_layer_q(second_encoder)
        latent_graph_encoder_v = self.latent_graph_layer_v(second_encoder)

        latent_graph = self.latent_graph_encoder_att(
            query=latent_graph_encoder_v,
            value=latent_graph_encoder_q,
            key=latent_graph_encoder_k,
            attention_mask=graph_mask,
        )
        return k_layers.Add()([latent_set, latent_graph])
