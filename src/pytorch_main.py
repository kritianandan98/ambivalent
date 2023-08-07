"""
File to train and test torch models.
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
from .dataset import AmbiDataset, train_collate_fn, test_collate_fn
from .models import Cnn14, Cnn10, Cnn6, CNN, MLP, LSTM, BiLSTM, Wav2Vec2
from .evaluate import Evaluator, calculate_accuracy, calculate_F1
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#from torchinfo import summary
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
    grad_accum = config.grad_accum
    num_workers = config.num_workers
    run_name = config.run_name
    classes_num = config.classes_num
    feature_name = config.feature_name

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
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
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    # Data loaders
    train_dataset = AmbiDataset(train_hdf5_path)
    val_dataset = AmbiDataset(val_hdf5_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_collate_fn, 
        num_workers=num_workers, pin_memory=True)
    print("Number of batches: ", len(train_loader))

    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=train_collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizers
    # Adam
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
    #    eps=1e-08, weight_decay=0., amsgrad=True)

    # SGD minibatch-gradient descent
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    #T_0 = 5
    #T_mult = 1
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)

    #if 'mixup' in augmentation:
    #    mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator
    evaluator = Evaluator(model=model)

    # Summary of the model
    #print(train_dataset[0]['features'].shape)
    #summary(model, input_size=(train_dataset[0]['features'].shape))

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
    if minidata:
        class_weights = np.ones(len(config.labels))
    class_weights = torch.tensor(class_weights,dtype=torch.float)
    class_weights = class_weights.to(device)

    train_loss_params = {"weights": class_weights, "total_epochs": n_epochs, "total_iterations": n_epochs * len(train_loader), "exp_base": 6, "transit_time_ratio": 0.25 , "counter": "iteration"}
    loss_func = get_loss_func(loss_type, train_loss_params)


    wandb.watch(model, log="all")

    best_valid_f1 = 0
    optimizer.zero_grad()

    epoch = 0
    iteration = 1
    while epoch < n_epochs:
        # Set model to train mode
        labels_changed = 0
        labels_corrected = 0
        model.train(True)

        n_batches = 0
        # Train on mini batches
        for batch_id, batch_data_dict in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

            #print("features", batch_data_dict['features'])
            #print("features", batch_data_dict['features'].shape)
            #print(batch_data_dict['features'].shape)
            #batch_data_dict['features'] = move_data_to_device(batch_data_dict['features'], device)
            batch_output_dict = {"clipwise_output": model(batch_data_dict['features'])}
            #"""{'clipwise_output': (batch_size, classes_num), ...}"""
            #print("target", batch_data_dict['target'].shape)
            #print("clipwise output", batch_output_dict['clipwise_output'].shape)
            batch_data_dict['target'] = move_data_to_device(batch_data_dict['target'], device)
            batch_target_dict = {'target': batch_data_dict['target']}
            #    """{'target': (batch_size, classes_num)}"""

            # loss
            #print("clip wise", batch_output_dict['clipwise_output'])
            #print(" target ", batch_target_dict['target'].shape)
            #print("total batches", len(train_loader))
            #print("total iter", loss_params['total_iterations'])
            #print("cur time", iteration)
            batch_loss, new_targets = loss_func(y_pred=batch_output_dict['clipwise_output'], y_true=batch_target_dict['target'], cur_time=iteration)
            if loss_type == 'proselflc':
                old_targets = torch.argmax(batch_data_dict['target'], -1)
                new_targets = torch.argmax(new_targets, -1)
                gt = torch.argmax(batch_data_dict['gt'], -1)
                #print(new_targets)
                #print(old_targets)
                #print(gt)
                for i, t in enumerate(new_targets):
                    if t != old_targets[i]:
                        if t == gt[i]:
                            labels_corrected += 1
                        labels_changed += 1


            ## Backward (mini batch GD)
            #print(batch_loss)
            batch_loss.backward() # calculate gradients

            if ((batch_id + 1) % grad_accum == 0) or (batch_id + 1 == len(train_loader)): # gradient accumulation
                optimizer.step()
                optimizer.zero_grad()

            n_batches += 1
            iteration += 1

            #scheduler.step(epoch + i / len(train_loader))


        ## Set the model to evaluation mode, disabling dropout and using population
        # Train Statistics
        wandb.log({'train labels corrected': labels_corrected, 'train labels changed': labels_changed}, step=epoch)
        model.eval()
        train_statistics, pred_labels = evaluator.evaluate(train_loader)
        logging.info('Train accuracy: {:.3f}'.format(train_statistics['accuracy']))
        logging.info('Train F1: {:.3f}'.format(train_statistics['f1_score']))

        logging.info('Epoch: {} Train Loss  : {:.3f}, '.format(epoch, train_statistics['cce-hard']))
        wandb.log({"train/loss": train_statistics['cce-hard'] ,"train/f1": train_statistics['f1_score']}, step=epoch)

        # Validation Statistics
        # val statistics
        val_statistics, pred_labels = evaluator.evaluate(validate_loader)
        logging.info('Validation loss: {:.3f}'.format(val_statistics['cce-hard']))
        logging.info('Validate accuracy: {:.3f}'.format(val_statistics['accuracy']))
        logging.info('Validate F1: {:.3f}'.format(val_statistics['f1_score']))
        print("pred labels: ", pred_labels)
        val_f1 = val_statistics['f1_score']

        wandb.log({"val/cce-hard": val_statistics['cce-hard'], "val/cce-soft": val_statistics['cce-soft'], "val/f1": val_statistics['f1_score'], "val/ece" : val_statistics["ece"], "val/brier-hard": val_statistics["brier-hard"], "val/brier-soft": val_statistics["brier-soft"], "val/kl-hard": val_statistics['kl-hard'], "val/kl-soft": val_statistics['kl-soft']}, step=epoch)

        scheduler.step(val_f1)

        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
            wandb.log({'lr': param_group['lr']}, step=epoch)

        ## save model if drop in valid loss
        if val_f1 > best_valid_f1 and os.environ["WANDB_MODE"] == "online":
            best_valid_f1 = val_f1
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
    
    #test_hdf5_path = train_hdf5_path = os.path.join(workspace, 'features', feature_name + '_' + 'val_waveform.h5')
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
    #model = torch.nn.DataParallel(model)

    dataset = AmbiDataset(test_hdf5_path)

    # Data Loader

    test_loader = torch.utils.data.DataLoader(dataset=dataset, collate_fn=test_collate_fn, batch_size=1,
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    loss_func = get_loss_func(loss_type='cce', params=None)
     
    model.eval()

    if 'Wav2Vec2' in run_name:
        # Load original pre-trained model and processor
        test_len = len(test_loader)
        test_len = 100
        x = torch.vstack([torch.Tensor(dataset[i]['features'][0]) for i in range(test_len)])
        x = x.to('cuda')

        predicted_embeddings = []
        labels = []
        for input in x:
            input = input.unsqueeze(0)
            embed = torch.mean(model.upstream(input)['last_hidden_state'], dim=1).squeeze()
            embed = embed.detach().cpu().numpy()
            #print(embed.shape)
            predicted_embeddings.append(embed)

        predicted_embeddings = np.array(predicted_embeddings)
        # Perform t-SNE dimensionality reduction
        print(predicted_embeddings.shape)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne.fit_transform(predicted_embeddings)

        # Plot the t-SNE embeddings
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], marker='o', color='b', s=10)
        plt.title("t-SNE Embeddings from Model Predictions")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)
        plt.savefig('/home/kriti/ambivalent/images/embed-plot.png')

    logging.info('Testing model..')

    outputs = []
    targets = []

    seg_outputs = []
    seg_targets = []
    for id, data_dict in tqdm(enumerate(test_loader), total=len(test_loader)):
        segment_preds = np.zeros(classes_num, dtype='float32')
        segment_preds = move_data_to_device(segment_preds, device)
        for seg_features in data_dict['features']:
            #print(seg_features.dtype)
            seg_features = move_data_to_device(seg_features, device)
            seg_features = seg_features.unsqueeze(0) # batch_size of 1
            #print("seg features", seg_features.shape)
            prediction = model(seg_features)[0]
            segment_preds += prediction
            seg_outputs.append(prediction.detach().cpu().numpy())
            seg_targets.append(data_dict['target'])
        segment_preds /= len(data_dict['features'])
        outputs.append(segment_preds.detach().cpu().numpy())
        targets.append(data_dict['target'])

    #print("outputs", outputs)
    #print("targets", targets)

    outputs = np.array(outputs, dtype='float32') # (audios_num, classes_num)
    targets = np.vstack(targets)

    seg_outputs = np.array(seg_outputs, dtype='float32')
    seg_targets = np.vstack(seg_targets)

    #print("output shape", outputs.shape)
    #print("targets shape", targets.shape)

    #print("logits:", outputs)
    predictions = outputs / sum(outputs)
    seg_predictions = seg_outputs / sum(seg_outputs)
    #print("target:", targets)

    cm = metrics.confusion_matrix(np.argmax(targets, axis=-1), np.argmax(predictions, axis=-1), labels=None)
    print(cm)

    report = metrics.classification_report(np.argmax(targets, -1), np.argmax(predictions, axis=-1), target_names=config.labels)
    print("Adding up predictions")
    print(report)

    report = metrics.classification_report(np.argmax(seg_targets, -1), np.argmax(seg_predictions, axis=-1), target_names=config.labels)
    print("Segment wise")
    print(report)

    print("targets", targets)
    print("predictions", predictions)
    accuracy = calculate_accuracy(targets, predictions)
    f1 = calculate_F1(targets, predictions)

    logging.info("Statistics on Noisy labels as ground truth:")
    logging.info('Test accuracy: {:.3f}'.format(accuracy))
    logging.info('Test F1: {:.3f}'.format(f1))

    logging.info('Storing predictions...')
    # store results in a CSV
    df = pd.DataFrame(columns=['wavfile', 'ground-truth', 'prediction', 'audio-path'])

    with h5py.File(test_hdf5_path, 'r') as hf:
        for i, audiopath in enumerate(hf['info']['audio_name']):
            df.loc[i, 'wavfile'] = audiopath
            df.loc[i, 'prediction'] = config.idx_to_lb[np.argmax(predictions[i], axis=-1)]
            df.loc[i, 'pred-conf'] = str(predictions[i])
            df.loc[i, 'ground-truth'] = config.idx_to_lb[np.argmax(hf['info']['gt'][i], axis=-1)] # store ground truth in 
            df.loc[i, 'gt-conf'] = str(hf['info']['gt'][i])
            df.loc[i, 'soft-gt-conf'] = str(hf['info']['soft-gt'][i])
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