from keras import layers as k_layers
from keras import models as k_models
from keras import regularizers as k_regularizers
import tensorflow as tf

from cellohood.models import utils


tf.compat.v1.disable_eager_execution()


class BaseEncoder(k_models.Model):

    def __init__(
            self,
            first_layer_size=128,
            second_layer_size=128,
            latent_size=64,
    ):
        super(BaseEncoder, self).__init__()
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.latent_size = latent_size

        self.first_encoder_layer = k_layers.Dense(self.first_layer_size)
        self.second_encoder_layer = k_layers.Dense(self.second_layer_size)
        self.latent_layer = k_layers.Dense(latent_size)

    def call(self, inputs):
        first_layer_output = self.first_encoder_layer(inputs)
        first_layer_output = k_layers.Activation('relu')(first_layer_output)

        second_layer_output = self.second_encoder_layer(first_layer_output)
        second_layer_output = k_layers.Activation('relu')(second_layer_output)

        latent = self.latent_layer(second_layer_output)

        return latent


class BaseVAEEncoder(k_models.Model):

    def __init__(
            self,
            first_layer_size=128,
            second_layer_size=128,
            latent_size=64,
            variance_activation=None,
            l1_weight=1e-4,
            l2_weight=1e-3,
    ):
        super(BaseVAEEncoder, self).__init__()
        self.first_layer_size = first_layer_size
        self.second_layer_size = second_layer_size
        self.latent_size = latent_size

        self.first_encoder_layer = k_layers.Dense(
            self.first_layer_size,
            kernel_regularizer=k_regularizers.L1L2(l1=l1_weight, l2=l2_weight),
        )
        self.second_encoder_layer = k_layers.Dense(self.second_layer_size)
        self.latent_mean_layer = k_layers.Dense(latent_size)
        self.latent_variance_layer = k_layers.Dense(latent_size)

        self.variance_activation = variance_activation if variance_activation is not None else\
            utils.base_variance_activation

    def call(self, inputs):
        if isinstance(inputs, list) and len(inputs) > 1:
            inputs = inputs[0]
        first_layer_output = self.first_encoder_layer(inputs)
        first_layer_output = k_layers.Activation('relu')(first_layer_output)

        second_layer_output = self.second_encoder_layer(first_layer_output)
        second_layer_output = k_layers.Activation('relu')(second_layer_output)

        latent_mean = self.latent_mean_layer(second_layer_output)
        latent_variance = self.latent_variance_layer(second_layer_output)
        latent_variance = k_layers.Activation(self.variance_activation)(latent_variance)

        return latent_mean, latent_variance


class BaseGNNEncoder(k_models.Model):

    def __init__(
            self,
            layer_size=128,
            intermediate_layer_size=256,
            num_heads=4,
            key_size=64,
            key_dim=32,
            latent_size=64,
    ):
        super(BaseGNNEncoder, self).__init__()

        self.encoding_layer = k_layers.Dense(layer_size)

        self.first_encoder_layer_k = k_layers.Dense(key_size)
        self.second_encoder_layer_k = k_layers.Dense(key_size)
        self.latent_layer_k = k_layers.Dense(key_size)

        self.first_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.second_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.latent_layer_v = k_layers.Dense(latent_size)

        self.first_encoder_layer_q = k_layers.Dense(key_size)
        self.second_encoder_layer_q = k_layers.Dense(key_size)
        self.latent_layer_q = k_layers.Dense(key_size)

        self.first_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.second_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.latent_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=latent_size,
        )
        self.first_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')
        self.second_encoder_layer_ff_1 = k_layers.Dense(intermediate_layer_size, activation='relu')

        self.first_encoder_layer_ff_2 = k_layers.Dense(layer_size)
        self.second_encoder_layer_ff_2 = k_layers.Dense(layer_size)

        self.first_layer_norm = k_layers.LayerNormalization()
        self.second_layer_norm = k_layers.LayerNormalization()

    def call(self, inputs):
        marker_input, cell_type_input, mask_adjacency = inputs[0], inputs[1], inputs[2]
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

        latent_encoder_k = self.latent_layer_k(second_encoder)
        latent_encoder_q = self.latent_layer_q(second_encoder)
        latent_encoder_v = self.latent_layer_v(second_encoder)

        latent = self.latent_encoder_att(
            query=latent_encoder_v,
            value=latent_encoder_q,
            key=latent_encoder_k,
            attention_mask=mask_adjacency,
        )
        final_output = k_layers.Lambda(lambda x: x[:, 0])(latent)
        return final_output


class BaseGNNVAEEncoder(k_models.Model):

    def __init__(
            self,
            layer_size=128,
            intermediate_layer_size=256,
            num_heads=4,
            key_size=64,
            key_dim=32,
            latent_size=64,
    ):
        super(BaseGNNVAEEncoder, self).__init__()

        self.encoding_layer = k_layers.Dense(layer_size)

        self.first_encoder_layer_k = k_layers.Dense(key_size)
        self.second_encoder_layer_k = k_layers.Dense(key_size)
        self.latent_mean_layer_k = k_layers.Dense(key_size)
        self.latent_std_layer_k = k_layers.Dense(key_size)

        self.first_encoder_layer_v = k_layers.Dense(layer_size)
        self.second_encoder_layer_v = k_layers.Dense(layer_size)
        self.latent_mean_layer_v = k_layers.Dense(latent_size)
        self.latent_std_layer_v = k_layers.Dense(latent_size)

        self.first_encoder_layer_q = k_layers.Dense(key_size)
        self.second_encoder_layer_q = k_layers.Dense(key_size)
        self.latent_mean_layer_q = k_layers.Dense(key_size)
        self.latent_std_layer_q = k_layers.Dense(key_size)

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
        self.latent_mean_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=latent_size,
        )
        self.latent_std_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=latent_size,
        )
        self.first_layer_norm = k_layers.LayerNormalization()
        self.second_layer_norm = k_layers.LayerNormalization()

    def call(self, inputs):
        marker_input, cell_type_input, mask_adjacency = inputs[0], inputs[1], inputs[2]
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

        latent_mean_encoder_k = self.latent_mean_layer_k(second_encoder)
        latent_mean_encoder_q = self.latent_mean_layer_q(second_encoder)
        latent_mean_encoder_v = self.latent_mean_layer_v(second_encoder)

        latent_std_encoder_k = self.latent_std_layer_k(second_encoder)
        latent_std_encoder_q = self.latent_std_layer_q(second_encoder)
        latent_std_encoder_v = self.latent_std_layer_v(second_encoder)

        latent_mean = self.latent_mean_encoder_att(
            query=latent_mean_encoder_v,
            value=latent_mean_encoder_q,
            key=latent_mean_encoder_k,
            attention_mask=mask_adjacency,
        )
        latent_std = self.latent_std_encoder_att(
            query=latent_std_encoder_v,
            value=latent_std_encoder_q,
            key=latent_std_encoder_k,
            attention_mask=mask_adjacency,
        )

        final_mean = k_layers.Lambda(lambda x: x[:, 0])(latent_mean)
        final_std = k_layers.Lambda(lambda x: x[:, 0])(latent_std)
        return final_mean, final_std


class FullOutputBaseGNNEncoder(k_models.Model):

    def __init__(
            self,
            layer_size=128,
            intermediate_layer_size=256,
            num_heads=4,
            key_size=64,
            key_dim=32,
            latent_size=64,
    ):
        super(FullOutputBaseGNNEncoder, self).__init__()
        self.latent_size = latent_size
        self.encoding_layer = k_layers.Dense(layer_size)

        self.first_encoder_layer_k = k_layers.Dense(key_size)
        self.second_encoder_layer_k = k_layers.Dense(key_size)
        self.latent_layer_k = k_layers.Dense(key_size)

        self.first_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.second_encoder_layer_v = k_layers.Dense(layer_size, activation='relu')
        self.latent_layer_v = k_layers.Dense(latent_size)

        self.first_encoder_layer_q = k_layers.Dense(key_size)
        self.second_encoder_layer_q = k_layers.Dense(key_size)
        self.latent_layer_q = k_layers.Dense(key_size)

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
        self.latent_encoder_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=latent_size,
        )
        self.first_layer_norm = k_layers.LayerNormalization()
        self.second_layer_norm = k_layers.LayerNormalization()

    def call(self, inputs):
        marker_input, cell_type_input, mask_adjacency = inputs[0], inputs[1], inputs[2]
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

        latent_encoder_k = self.latent_layer_k(second_encoder)
        latent_encoder_q = self.latent_layer_q(second_encoder)
        latent_encoder_v = self.latent_layer_v(second_encoder)

        latent = self.latent_encoder_att(
            query=latent_encoder_v,
            value=latent_encoder_q,
            key=latent_encoder_k,
            attention_mask=mask_adjacency,
        )
        return latent

class FullOutputBaseGNNSetAndGraphEncoder(k_models.Model):

    def __init__(
            self,
            layer_size=128,
            intermediate_layer_size=256,
            num_heads=4,
            key_size=64,
            key_dim=32,
            latent_size=64,
    ):
        super(FullOutputBaseGNNSetAndGraphEncoder, self).__init__()
        self.latent_size = latent_size
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
        self.latent_set_layer_v = k_layers.Dense(latent_size)
        self.latent_graph_layer_v = k_layers.Dense(latent_size)

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

