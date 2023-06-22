"""
File to train and test the model.
"""
import os
import wandb
import h5py
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
from . import config
 
from .losses import get_loss_func
from .utils import move_data_to_device, do_mixup
from .utils import (create_folder, get_filename, create_logging)
from .dataset import AmbiDataset, collate_fn
from .models import Cnn14, Cnn10, Cnn6, CNN, DNN, LSTM
from .evaluate import Evaluator
from sklearn.utils.class_weight import compute_class_weight

from torchinfo import summary
#from torchsummary import summary

os.environ["WANDB_MODE"] = "online"


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    filename = args.filename
    resume_epoch = args.resume_epoch
    minidata = args.minidata

    run_name = config.run_name
    model_type = config.model_type
    pretrained_checkpoint_path = config.pretrained_checkpoint_path
    freeze_base = config.freeze_base
    loss_type = config.loss_type
    n_epochs = config.epochs
    augmentation = config.augmentation
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    num_workers = config.num_workers
    run_name = config.run_name
    classes_num = config.classes_num
    feature_name = config.feature_name

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    
    if minidata:
        train_hdf5_path = os.path.join(workspace, 'features', feature_name + '_' + 'train_minidata_waveform.h5')
        val_hdf5_path = os.path.join(workspace, 'features',  feature_name + '_' + 'val_minidata_waveform.h5')
    else:
        train_hdf5_path = os.path.join(workspace, 'features',  feature_name + '_' + 'train_waveform.h5')
        val_hdf5_path = os.path.join(workspace, 'features',  feature_name + '_' + 'val_waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, model_type, '{}'.format(run_name), 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
         'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)
    
    logs_dir = os.path.join(workspace, 'logs', filename, model_type, '{}'.format(run_name), 'pretrain={}'.format(pretrain), 
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
    model = Model(classes_num)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_epoch:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_epoch.pth'.format(resume_epoch))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        epoch = resume_checkpoint['epoch']
    else:
        epoch = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    # Data loaders
    train_dataset = AmbiDataset(train_hdf5_path)
    val_dataset = AmbiDataset(val_hdf5_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizers
    # Adam
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
    #    eps=1e-08, weight_decay=0., amsgrad=True)

    # SGD minibatch-gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    #T_0 = 5
    #T_mult = 1
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    #if 'mixup' in augmentation:
    #    mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator
    evaluator = Evaluator(model=model)

    # Summary of the model
    summary(model, input_size=(train_dataset[0]['features'].shape))

    # Use wandb if enabled
    wandb.init(
        # set the wandb project where this run will be logged
        project="ambivalent",

        name=run_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": model_type,
        "dataset": "IEMOCAP",
        "epochs": n_epochs,
        "batch_size": batch_size,
        }
    )

    # class weights
    with h5py.File(train_hdf5_path, 'r') as hf:
        targets = np.argmax(np.array(hf['info']['target'][:]), axis=-1)
    #print(targets)

    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(targets),y=targets)
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    class_weights = class_weights.to(device)

    wandb.watch(model, log="all")

    best_valid_loss=float('inf')
    
    while epoch < n_epochs:
        # Set model to train mode
        model.train(True)

        n_batches = 0
        # Train on mini batches
        for i, batch_data_dict in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            # Move data to GPU
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            #print("features", batch_data_dict['features'])

            batch_output_dict = {"clipwise_output": model(batch_data_dict['features'])}
            #"""{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            #    """{'target': (batch_size, classes_num)}"""

            # loss
            batch_loss = loss_func(batch_output_dict['clipwise_output'], batch_target_dict['target'], class_weights)

            ## Backward (mini batch GD)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            n_batches += 1

            #scheduler.step(epoch + i / len(train_loader))


        ## Set the model to evaluation mode, disabling dropout and using population
        # Train Statistics
        train_statistics, pred_labels, train_loss = evaluator.evaluate(train_loader, loss_func, class_weights)
        logging.info('Train accuracy: {:.3f}'.format(train_statistics['accuracy']))
        logging.info('Train F1: {:.3f}'.format(train_statistics['f1_score']))

        logging.info('Epoch: {} Train Loss  : {:.3f}, '.format(epoch, train_loss))
        wandb.log({"train/loss": train_loss ,"train/f1": train_statistics['f1_score']}, step=epoch)

        # Validation Statistics
        if epoch % 1 == 0:
            model.eval()
            # val statistics
            val_statistics, pred_labels, val_loss = evaluator.evaluate(validate_loader, loss_func, class_weights)
            logging.info('Validation loss: {:.3f}'.format(val_loss))
            logging.info('Validate accuracy: {:.3f}'.format(val_statistics['accuracy']))
            logging.info('Validate F1: {:.3f}'.format(val_statistics['f1_score']))
            print("pred labels: ", pred_labels)

            wandb.log({"val/loss": val_loss, "val/f1": val_statistics['f1_score']}, step=epoch)

        #scheduler.step(val_loss)

        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
            wandb.log({'lr': param_group['lr']}, step=epoch)

        ## save model if drop in valid loss
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            checkpoint = {
                'epoch': epoch, 
                'model': model.module.state_dict()}
#
            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_epoch.pth'.format(epoch))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

            checkpoint_path = os.path.join(
                checkpoints_dir, '_best.pth')
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        epoch += 1 # update epoch



def test(args):
    # Arugments & parameters
    filename = args.filename
    workspace = args.workspace
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename

    run_name = config.run_name
    model_type = config.model_type
    pretrained_checkpoint_path = config.pretrained_checkpoint_path
    freeze_base = config.freeze_base
    loss_type = config.loss_type
    augmentation = config.augmentation
    batch_size = config.batch_size
    num_workers = config.num_workers
    run_name = config.run_name
    classes_num = config.classes_num
    feature_name = config.feature_name

    pretrain = True if pretrained_checkpoint_path else False
    
    test_hdf5_path = train_hdf5_path = os.path.join(workspace, 'features', feature_name + '_' + 'test_waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, model_type, '{}'.format(run_name), 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
         'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    
    logs_dir = os.path.join(workspace, 'logs', filename, model_type, '{}'.format(run_name), 'pretrain={}'.format(pretrain), 
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
    model = Model(classes_num)


    checkpoint_path = os.path.join(checkpoints_dir, '_best.pth')
    logging.info('Load model from {}'.format(checkpoint_path))
    model_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(model_checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = AmbiDataset(test_hdf5_path)

    # Data Loader

    test_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=collate_fn, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    loss_func = get_loss_func(loss_type)
     
    # Evaluator
    evaluator = Evaluator(model=model)
    model.eval()

    logging.info('Testing model..')
    
    statistics, predictions, _ = evaluator.evaluate(test_loader, loss_func=loss_func)
    logging.info('Test accuracy: {:.3f}'.format(statistics['accuracy']))
    logging.info('Test F1: {:.3f}'.format(statistics['f1_score']))

    logging.info('Storing predictions...')
    # store results in a CSV
    df = pd.DataFrame(columns=['wavfile', 'ground-truth', 'prediction', 'audio-path'])

    with h5py.File(test_hdf5_path, 'r') as hf:
        for i, audiopath in enumerate(hf['info']['audio_name']):
            df.loc[i, 'wavfile'] = audiopath
            df.loc[i, 'prediction'] = config.idx_to_lb[predictions[i]]
            df.loc[i, 'ground-truth'] = config.idx_to_lb[np.argmax(hf['info']['gt'][i], axis=-1)]
            df.loc[i, 'audio-path'] = hf['info']['audio_path'][i]
        
    df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--resume_epoch', type=int)
    parser_train.add_argument('--minidata', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
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