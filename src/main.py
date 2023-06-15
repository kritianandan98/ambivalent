"""
File to train and test the model
"""
import os
import h5py
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
 
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size, 
    hop_size, idx_to_lb)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import AmbiDataset, TrainSampler, EvaluateSampler, TestSampler, collate_fn
from models import Cnn14, Cnn10, Cnn6
from evaluate import Evaluator


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    n_epochs = args.epochs
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_epoch = args.resume_epoch
    minidata = args.minidata
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    
    if minidata:
        hdf5_path = os.path.join(workspace, 'features', 'minidata_waveform.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
         'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    logging.info(f'Loading model: {model_type}, classes: {classes_num}')
    # Model
    Model = eval(model_type)
    #model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
    #    classes_num, freeze_base)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num)

    # Statistics
    # statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_epoch:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_epoch.pth'.format(resume_epoch))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        # statistics_container.load_state_dict(resume_epoch)
        epoch = resume_checkpoint['epoch']
    else:
        epoch = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = AmbiDataset()

    # Data generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizers
    # Adam
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
    #    eps=1e-08, weight_decay=0., amsgrad=True)

    # SGD minibatch-gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator
    evaluator = Evaluator(model=model)

    avg_loss = 0

    def param_count(module: nn.Module) -> int:
        return sum([p.data.nelement() for p in module.parameters()])
    
    logging.info(f"Model Parameter count {param_count(model)}")
    
    while epoch < n_epochs:
        epoch_loss = 0
        # Set model to train mode
        model.train(True)

        n_batches = 0
        # Train on mini batches
        for batch_data_dict in tqdm(train_loader, total=len(train_loader)):
            
            if 'mixup' in augmentation:
                batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))
            
            # Move data to GPU
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            if 'mixup' in augmentation:
                batch_output_dict = model(batch_data_dict['logmel_feat'], 
                    batch_data_dict['mixup_lambda'])
                """{'clipwise_output': (batch_size, classes_num), ...}"""
            
                batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                    batch_data_dict['mixup_lambda'])}
                """{'target': (batch_size, classes_num)}"""
            else:
                batch_output_dict = model(batch_data_dict['logmel_feat'], None)
            #"""{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            #    """{'target': (batch_size, classes_num)}"""

            # loss
            loss = loss_func(batch_output_dict, batch_target_dict)
            epoch_loss += loss.item()

            ## Backward (mini batch GD)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batches += 1

        epoch_loss /= n_batches
        logging.info('Epoch: {} Train Loss  : {:.3f}, '.format(epoch, epoch_loss))

        ## Set the model to evaluation mode, disabling dropout and using population
        ## statistics for batch normalization. Evaluate every 10 epochs
        if epoch % 1 == 0:
            model.eval()
            # train statistics
            statistics, pred_labels = evaluator.evaluate(train_loader)
            logging.info('Train accuracy: {:.3f}'.format(statistics['accuracy']))
            logging.info('Train F1: {:.3f}'.format(statistics['f1_score']))
            # val statistics
            statistics, pred_labels = evaluator.evaluate(validate_loader)
            logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))
            logging.info('Validate F1: {:.3f}'.format(statistics['f1_score']))
            print("pred labels: ", pred_labels)
        #    #statistics_container.append(epoch, statistics, 'validate')
        #    #statistics_container.dump()

        ## save model checkpoints every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch, 
                'model': model.module.state_dict()}
#
            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_epoch.pth'.format(epoch))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
#
        avg_loss += epoch_loss # collect batch loss
        epoch += 1 # update epoch

    avg_loss /= n_epochs

    print(f"End Train loss {avg_loss}") 



def test(args):
    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    model_checkpoint_path = args.model_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = 'none'
    batch_size = args.batch_size
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8 

    pretrain = False
    
    hdf5_path = os.path.join(workspace, 'features', 'test-waveform.h5')
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    logging.info(f"Using model: {model_type}, classes: {classes_num}")
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)

    logging.info('Load model from {}'.format(model_checkpoint_path))
    model_checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(model_checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = AmbiDataset()

    # Data generator

    test_sampler = TestSampler(
        hdf5_path=hdf5_path, 
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)
     
    # Evaluator
    evaluator = Evaluator(model=model)
    model.eval()

    logging.info('Testing model..')
    
    statistics, predictions = evaluator.evaluate(test_loader)
    logging.info('Test accuracy: {:.3f}'.format(statistics['accuracy']))
    logging.info('Test F1: {:.3f}'.format(statistics['f1_score']))

    logging.info('Storing predictions...')
    # store results in a CSV
    df = pd.DataFrame(columns=['wavfile', 'ground-truth', 'prediction', 'audio-path'])

    with h5py.File(hdf5_path, 'r') as hf:
        for i, audiopath in enumerate(hf['audio_name']):
            df.loc[i, 'wavfile'] = audiopath
            df.loc[i, 'prediction'] = idx_to_lb[predictions[i]]
            df.loc[i, 'ground-truth'] = idx_to_lb[np.argmax(hf['gt'][i], axis=-1)]
            df.loc[i, 'audio-path'] = hf['audio_path'][i]
        
    df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--epochs', type=int, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_epoch', type=int)
    parser_train.add_argument('--minidata', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_test.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_test.add_argument('--model_type', type=str, required=True)
    parser_test.add_argument('--model_path', type=str, required=True)
    parser_test.add_argument('--freeze_base', action='store_true', default=False)
    parser_test.add_argument('--loss_type', type=str, required=True)
    parser_test.add_argument('--learning_rate', type=float)
    parser_test.add_argument('--batch_size', type=int, required=True)
    parser_test.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise Exception('Error argument!')