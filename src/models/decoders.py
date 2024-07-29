from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import get_linear_layers

from torch.distributions.multinomial import Multinomial


class Decoder(nn.Module, ABC):

    def __init__(self, num_samples, output_size):
        """ Decoder parent class with no specified output distribution
        """
        super(Decoder, self).__init__()
        self.num_samples = num_samples
        self.output_size = output_size

    def get_probs(self, logits, is_ensemble):
        if is_ensemble:
            predictions_per_sample = torch.argmax(logits, -1)
            pre_instance_occurrences = F.one_hot(predictions_per_sample, self.output_size).sum(1)
            probs = pre_instance_occurrences / self.num_samples
        else:
            logits = logits.mean(1)
            probs = F.softmax(logits, dim=-1)
        return probs

    @abstractmethod
    def __call__(self, x, is_train=False):
        pass


# region Multinomial

class MultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 num_samples,
                 output_size,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3
                 ):
        super(MultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def __call__(self, z, is_ensemble=False):
        logits = self.net(z)
        probs = self.get_probs(logits, is_ensemble)
        return Multinomial(probs=probs)


class FlattenMultinomialDecoder(Decoder):
    """
    Uses MLP layers to calculate the logits of the multinomial distribution of p(y|z).
    z is of shape z_dim X time_length which means that z is a matrix.
    The MLP runs over the last therefore we flatten the matrix which means that the first MLP size is of
    z_dim X time_length.
    """

    def __init__(self,
                 z_dim,
                 z_dim_time_length,
                 output_size,
                 num_samples,
                 hidden_size_1,
                 hidden_size_2,
                 hidden_size_3,
                 **kwargs):
        super(FlattenMultinomialDecoder, self).__init__(num_samples, output_size)
        input_size = z_dim * z_dim_time_length
        hidden_sizes = [input_size]
        for hidden_size in [hidden_size_1, hidden_size_2, hidden_size_3]:
            if hidden_size == -1:
                break
            hidden_sizes.append(hidden_size)
        hidden_sizes.append(output_size)
        layers = get_linear_layers(hidden_sizes)
        self.net = nn.Sequential(*layers)

    def _get_logits(self, z):
        z = z.flatten(-2)
        logits = self.net(z)
        logits = logits.reshape(-1, z.shape[1], logits.shape[-1])
        return logits

    def __call__(self, z, is_ensemble=False):
        logits = self._get_logits(z)
        probs = self.get_probs(logits, is_ensemble)

        return Multinomial(probs=probs)

# endregion
