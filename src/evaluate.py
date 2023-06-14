import numpy as np
import logging
import torch.nn.functional as F
from sklearn import metrics
from scipy.special import softmax

from pytorch_utils import forward
from utilities import get_filename
import config


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
        target = output_dict['target']    # (audios_num, classes_num)
        #print("predictions:", predictions)

        cm = metrics.confusion_matrix(np.argmax(target, axis=-1), np.argmax(predictions, axis=-1), labels=None)
        accuracy = calculate_accuracy(target, predictions)
        f1 = calculate_F1(target, predictions)

        statistics = {'accuracy': accuracy, 'f1_score': f1}
        pred_labels = np.argmax(predictions, axis=-1)
        
        return statistics, pred_labels
