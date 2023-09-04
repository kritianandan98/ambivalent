"""
Loss functions
"""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ParamException(Exception):
    """
    Invalid parameter exception
    """

    def __init__(self, msg, fields=None):
        self.fields = fields
        self.msg = msg

    def __str__(self):
        return self.msg


def get_loss_func(loss_type: str, params=None):
    if loss_type == "cce":
        return CrossEntropy()
    if loss_type == "kl":
        return KLDiv()
    elif loss_type == "proselflc":
        return ProSelfLC(params)
    else:
        print("Incorrect loss fn name!")


class KLDiv(nn.Module):
    """
    Compute the Kullback-Leibler (KL) Divergence Loss given predicted probabilities and target probabilities.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, cur_time: int) -> Tensor:
        """
        Arguments
            y_pred: PyTorch tensor of shape (num_samples, num_classes) representing predicted probabilities.
            y_true: PyTorch tensor of shape (num_samples, num_classes) representing target probabilities.

        Returns
            loss: scalar value representing the KL Divergence loss averaged over the samples.
        """
        assert (
            y_pred.shape == y_true.shape
        ), "Shapes of y_pred and y_true must be the same."

        # Compute the KL Divergence Loss using the PyTorch function kl_div
        loss = F.kl_div(y_pred.log(), y_true, reduction="mean")

        return loss


class CrossEntropy(nn.Module):
    """
    The new implementation of cross entropy using two distributions.
    This can be a base class for other losses:
        1. label smoothing;
        2. bootsoft (self label correction), joint-soft,etc.
        3. proselflc
        ...
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor, cur_time: int) -> Tensor:
        """
        Arguments
            y_pred: PyTorch tensor of shape (num_samples, num_classes) representing predicted probabilities.
            y_true: PyTorch tensor of shape (num_samples, num_classes) representing target probabilities.

        Returns
            loss: scalar value representing the Cross Entropy loss averaged over the samples.
        """
        epsilon = 1e-15  # Small constant to avoid log(0)

        # Ensure y_pred and y_true have the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "Shapes of y_pred and y_true must be the same."

        # Clip the predicted probabilities to avoid log(0)
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Compute the Cross Entropy Loss
        loss = -torch.sum(y_true * torch.log(y_pred), dim=1)
        avg_loss = torch.mean(loss)

        return avg_loss, None


class ProSelfLC(CrossEntropy):
    """
    The implementation for progressive self label correction (CVPR 2021 paper).
    The target probability will be corrected by
    a predicted distributions, i.e., self knowledge.
        1. ProSelfLC is partially inspired by prior related work,
            e.g., Pesudo-labelling.
        2. ProSelfLC is partially theorectically bounded by
            early stopping regularisation.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. current time (epoch/iteration counter).
        4. total time (total epochs/iterations)
        5. exp_base: the exponential base for adjusting epsilon
        6. counter: iteration or epoch counter versus total time.

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(
        self,
        params: dict = None,
    ) -> None:
        super().__init__()
        self.total_epochs = params["total_epochs"]
        self.exp_base = params["exp_base"]
        self.counter = params["counter"]
        self.epsilon = None
        self.transit_time_ratio = params["transit_time_ratio"]

        if not (self.exp_base >= 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero. "
            )
            raise (ParamException(error_msg))

        if not (isinstance(self.total_epochs, int) and self.total_epochs > 0):
            error_msg = (
                "self.total_epochs = "
                + str(self.total_epochs)
                + ". "
                + "The total_epochs has to be a positive integer. "
            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))

        if "total_iterations" in params.keys():
            # only exist when counter == "iteration"
            self.total_iterations = params["total_iterations"]

    def update_epsilon_progressive_adaptive(self, pred_probs: Tensor, cur_time: int):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )
            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[1]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 1
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            example_trust.to("cuda")
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]
            self.epsilon = self.epsilon.to("cuda")

    def forward(self, y_pred: Tensor, y_true: Tensor, cur_time: int) -> Tensor:
        """
        Args
            y_pred      : predicted probability distributions of shape (N, C)
            y_true      : target probability  distributions of shape (N, C)
            cur_time    : current time (epoch/iteration counter).

        Returns
            Loss        : a scalar tensor, normalised by N.
        """
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        # update self.epsilon
        self.update_epsilon_progressive_adaptive(y_pred, cur_time)
        new_target_probs = (1 - self.epsilon) * y_true + self.epsilon * y_pred
        # reuse CrossEntropy's forward computation
        loss, _ = super().forward(y_pred, new_target_probs, cur_time)
        return loss, new_target_probs
