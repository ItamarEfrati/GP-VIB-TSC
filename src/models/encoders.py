from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.model_utils import get_linear_layers, get_1d_cnn_layers, Permute, get_cnn1d_output_dim, get_parameters_list


class Encoder(nn.Module, ABC):
    def __init__(self, encoding_size):
        """ Decoder parent class with no specified output distribution
            :param encoding_size: latent space dimensionality
        """
        super(Encoder, self).__init__()
        self.encoding_size = encoding_size

    @abstractmethod
    def __call__(self, x):
        pass


# region Variational Inference
class BandedJointEncoder(Encoder):
    def __init__(self,
                 encoding_size,
                 z_dim_time_length,
                 precision_activation,
                 ):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            :param encoding_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param kernel_size: kernel size for Conv1D layer
        """
        super(BandedJointEncoder, self).__init__(encoding_size)
        self.precision_activation = precision_activation
        self.z_dim_time_length = z_dim_time_length

    def _get_sparse_matrix_indices(self, batch_size, sequence_length):
        num_variational_parameters = (2 * sequence_length - 1)
        first_dim = np.repeat(np.arange(batch_size), self.encoding_size * num_variational_parameters)
        second_dim = np.tile(np.repeat(np.arange(self.encoding_size), num_variational_parameters), batch_size)
        third_dim = np.tile(np.concatenate([np.arange(sequence_length), np.arange(sequence_length - 1)]),
                            batch_size * self.encoding_size)
        forth_dim = np.tile(np.concatenate([np.arange(sequence_length), np.arange(1, sequence_length)]),
                            batch_size * self.encoding_size)

        return np.stack([first_dim, second_dim, third_dim, forth_dim])

    def _get_covariance_matrix(self, precision_parameters, batch_size, sequence_length):
        sparse_matrix_indices = self._get_sparse_matrix_indices(batch_size, sequence_length)

        # There are 2T parameters for each sequence. Taking only 2T -1
        precision_parameters = precision_parameters[:, :, :-1].reshape(-1)
        sparse_matrix = torch.sparse_coo_tensor(sparse_matrix_indices, precision_parameters,
                                                (batch_size, self.encoding_size, sequence_length,
                                                 sequence_length))
        precision_tridiagonal = sparse_matrix.to_dense()

        batch_eye_matrix = torch.eye(sequence_length).reshape(1, 1, sequence_length, sequence_length).to(
            precision_parameters.device)
        batch_eye_matrix = batch_eye_matrix.repeat(batch_size, self.encoding_size, 1, 1)

        precision_tridiagonal += batch_eye_matrix
        # inverse of precision in covariance, precision_tridiagonal is upper tridiagonal
        covariance_upper_tridiagonal = torch.triangular_solve(batch_eye_matrix, precision_tridiagonal).solution
        covariance_upper_tridiagonal = torch.where(torch.isfinite(covariance_upper_tridiagonal),
                                                   covariance_upper_tridiagonal,
                                                   torch.zeros_like(covariance_upper_tridiagonal))

        return covariance_upper_tridiagonal

    def __call__(self, statistics):
        batch_size = statistics.shape[0]
        mu = statistics[:, :self.encoding_size]
        precision_parameters = statistics[:, self.encoding_size:]

        if self.precision_activation:
            precision_parameters = self.precision_activation(precision_parameters)
        precision_parameters = precision_parameters.reshape(batch_size, self.encoding_size, 2 * self.z_dim_time_length)

        covariance_upper_tridiagonal = self._get_covariance_matrix(precision_parameters, batch_size,
                                                                   self.z_dim_time_length)
        covariance_lower_tridiagonal = torch.permute(covariance_upper_tridiagonal, dims=(0, 1, 3, 2))

        return MultivariateNormal(loc=mu, scale_tril=covariance_lower_tridiagonal)


# endregion

# region CNN Time Series Encoders

class TimeSeriesDataEncoder(Encoder):
    """
    Extension to the classic with more layers and options
    """

    def __init__(self,
                 input_size,
                 ts_embedding_size,
                 kernel_size_1,
                 kernel_size_2,
                 kernel_size_3,
                 kernel_size_4,
                 padding_1,
                 padding_2,
                 padding_3,
                 padding_4,
                 out_channels_1,
                 out_channels_2,
                 out_channels_3,
                 out_channels_4,
                 dropout_1,
                 dropout_2,
                 dropout_3,
                 dropout_4,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 timeseries_size,
                 n_cnn_layers,
                 encoding_size
                 ):
        super().__init__(encoding_size)
        layers = get_linear_layers([input_size, ts_embedding_size]) if input_size != -1 else []
        layers += [Permute((0, 2, 1))]

        in_channels = ts_embedding_size

        cnn1d_sizes = get_parameters_list(in_channels, out_channels_1,
                                          out_channels_2, out_channels_3, out_channels_4,
                                          length=n_cnn_layers + 1)
        kernel_sizes = get_parameters_list(kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4,
                                           length=n_cnn_layers)
        padding = get_parameters_list(padding_1, padding_2, padding_3, padding_4,
                                      length=n_cnn_layers)

        dropout = get_parameters_list(dropout_1, dropout_2, dropout_3, dropout_4, length=n_cnn_layers)

        layers += get_1d_cnn_layers(cnn1d_sizes, kernel_sizes, padding, dropout)

        layers += [Permute((0, 2, 1))]

        self.encoding_series_size = get_cnn1d_output_dim(timeseries_size, kernel_sizes, padding)

        input_size = cnn1d_sizes[-1]
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)

        layers += get_linear_layers(hidden_sizes + [3 * encoding_size])
        layers += [Permute((0, 2, 1))]
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


# region Inception Time Series Encoders

class InceptionModule(nn.Module):
    def __init__(self, in_filters, nb_filters, kernel_size, use_bottleneck, bottleneck_size):
        super(InceptionModule, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size

        if use_bottleneck:
            self.input_inception = nn.Conv1d(in_channels=in_filters, out_channels=bottleneck_size, kernel_size=1,
                                             bias=False)
        else:
            bottleneck_size = in_filters

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        self.conv_list = nn.ModuleList()
        for k_size in kernel_size_s:
            self.conv_list.append(
                nn.Conv1d(in_channels=bottleneck_size, out_channels=nb_filters, kernel_size=k_size, stride=1,
                          padding='same', bias=False))

        self.max_pool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv1d(in_channels=in_filters, out_channels=nb_filters, kernel_size=1, bias=False)

        self.batch_norm = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x):
        if self.use_bottleneck and x.size(-1) > 1:
            input_inception = self.input_inception(x)
        else:
            input_inception = x

        conv_list = [conv(input_inception) for conv in self.conv_list]
        conv_list.append(self.conv_6(self.max_pool_1(x)))

        x = torch.cat(conv_list, dim=1)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class InceptionEncoder(Encoder):
    def __init__(self,
                 input_n_channels,
                 encoding_size,
                 encoding_series_size,
                 number_of_filters,
                 bottleneck_size,
                 use_bottleneck,
                 kernel_size=40,
                 depth=1
                 ):
        super().__init__(encoding_size)
        self.input_n_channels = input_n_channels
        self.encoding_size = encoding_size
        self.encoding_series_size = encoding_series_size
        self.number_of_filters = number_of_filters
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck
        self.kernels = [kernel_size, kernel_size // 2, kernel_size // 4]
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            current_block = self._get_next_block(d)
            self.blocks.append(current_block)
            residual_block = self._get_next_residual_block(d)
            self.residual_blocks.append(residual_block)

        self.last_block = nn.Sequential(
            nn.Conv1d(in_channels=4 * self.number_of_filters, out_channels=3 * self.encoding_size, kernel_size=1,
                      bias=False),
            nn.AdaptiveMaxPool1d(self.encoding_series_size),
        )

    def _get_next_residual_block(self, d):
        residual_block = nn.Sequential(Permute((0, 2, 1))) if d == 0 else nn.Sequential()
        in_channels = self.input_n_channels if d == 0 else 4 * self.number_of_filters
        residual_block.add_module(f'residual {d + 1} CNN',
                                  nn.Conv1d(in_channels=in_channels, out_channels=4 * self.number_of_filters,
                                            kernel_size=1,
                                            padding=0))
        residual_block.add_module(f'residual {d + 1} batch norm', nn.BatchNorm1d(4 * self.number_of_filters))
        return residual_block

    def _get_next_block(self, d):
        is_first_block = d == 0
        current_block = nn.Sequential(Permute((0, 2, 1))) if is_first_block else nn.Sequential()
        for i in range(3):
            in_size = self.input_n_channels if i == 0 and is_first_block else 4 * self.number_of_filters
            use_bottleneck = self.use_bottleneck if i == 0 else True
            inception_module = InceptionModule(in_size, self.number_of_filters, self.kernels[i],
                                               use_bottleneck=use_bottleneck,
                                               bottleneck_size=self.bottleneck_size)
            current_block.add_module(f'inception_module_{i + 1}', inception_module)
        return current_block

    def __call__(self, x):
        residual_input = x
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            residual = self.residual_blocks[i](residual_input)
            residual_input = x
            x = F.relu(x + residual)
        x = self.last_block(x)
        return x

# endregion
