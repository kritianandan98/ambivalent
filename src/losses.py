"""
Loss functions
"""
import torch.nn.functional as F


def cce(output_dict, target_dict):
    """
    Categorical cross entropy loss
    """
    return F.cross_entropy(output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'cce':
        return cce