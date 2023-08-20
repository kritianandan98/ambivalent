import torch
import numpy as np
import logging
import torch.nn.functional as F
from sklearn import metrics
from scipy.special import softmax
from . import config
from tqdm import tqdm
from .losses import get_loss_func

from .utils import forward
from .utils import move_data_to_device, expected_calibration_error_multiclass, brier_score_multiclass

def calculate_accuracy(y_true, y_score):
    accuracy = metrics.accuracy_score(np.argmax(y_true, axis=-1), np.argmax(y_score, axis=-1))
    return accuracy

def calculate_F1(y_true, y_score):
    F1 = metrics.f1_score(np.argmax(y_true, axis=-1), np.argmax(y_score, axis=-1), average='weighted')
    return F1


class Evaluator(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        # Forward
        output_dict = forward(
        model=self.model, 
        generator=data_loader, 
        return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        #print("logits:", clipwise_output)
        predictions = softmax(clipwise_output, axis=-1)
        targets = output_dict['target']    # (audios_num, classes_num)
        soft_targets = output_dict['soft-gt']

        ece = expected_calibration_error_multiclass(y_true=np.argmax(targets, -1), y_prob=predictions)
        brier_hard = brier_score_multiclass(y_true=targets, y_prob=predictions)
        brier_soft = brier_score_multiclass(y_true=soft_targets, y_prob=predictions)

        cm = metrics.confusion_matrix(np.argmax(targets, axis=-1), np.argmax(predictions, axis=-1), labels=None)
        clipwise_output_cuda = torch.FloatTensor(clipwise_output).to('cuda')
        targets_cuda = torch.FloatTensor(targets).to('cuda')
        soft_targets_cuda = torch.FloatTensor(soft_targets).to('cuda')

        cce_loss = get_loss_func('cce')
        kl_loss = get_loss_func('kl')

        cce_hard, _ = cce_loss(clipwise_output_cuda, targets_cuda, None)
        cce_soft, _ = cce_loss(clipwise_output_cuda, soft_targets_cuda, None)

        kl_hard = kl_loss(clipwise_output_cuda, targets_cuda, None)
        kl_soft = kl_loss(clipwise_output_cuda, soft_targets_cuda, None)            

        print("loss:", cce_hard)
        print("soft-loss:", cce_soft)
        #print("targets", targets)
        #print("predictions", predictions)
        accuracy = calculate_accuracy(targets, predictions)
        f1 = calculate_F1(targets, predictions)

        statistics = {'accuracy': accuracy, 'f1_score': f1, 'cce-hard': cce_hard, 'cce-soft': cce_soft, 'ece': ece, 'brier-hard': brier_hard, 'brier-soft': brier_soft, 'kl-hard': kl_hard, 'kl-soft': kl_soft}
        pred_labels = np.argmax(predictions, axis=-1)
        
        return statistics, pred_labels
