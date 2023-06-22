"""
Loss functions
"""
import torch.nn.functional as F


def cce(clipwise_output, target, weights=None):
    """
    Categorical cross entropy loss
    """
    return F.cross_entropy(input=clipwise_output, target=target, weight=weights)


def get_loss_func(loss_type):
    if loss_type == 'cce':
        return cce