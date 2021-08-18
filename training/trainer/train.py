""""
This script performs the training of the model that performs
Singing Language Identification. It is designed to submit a job
in Google Cloud AI Platform so that multiple machines
with GPUs can compute the necessary optimization algorithms.
"""


import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim, no_grad
from torch.utils.tensorboard import SummaryWriter
from tensorflow.python.lib.io import file_io

from .models import TCN_SLID
from .utils import training_output_string, data_generator_training, \
                        lang2idx, idx2lang, dict2str, split_train_test_valid, \
                        compute_accuracy
from .metrics import save_confusion_matrix, fig_titles

from collections import defaultdict


import stacklogging, logging


def step_train(model, x, y, criterion, optimizer):
    """
    Performs forward-backward pass, loss and update of the model parameters.
    (Applied to training dataset)

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        x : torch.Tensor
            The minibatch of training datapoints
        y: torch.Tensor
            The minibatch of training labels
        criterion : torch.nn.Module
            The loss criterion
        optimizer: torch.optim
            The optimization algorithm
    """

    # set model to training mode
    model.train()
    # forward pass
    y_pred = model(x)
    # compute loss
    loss = criterion(y_pred, y)
    # since the backward() method accumulates gradients and we
    # do not want to mix up gradients between minibatches, we zero them
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    # update model parameters
    optimizer.step()

    return y_pred, loss

@no_grad()
def step_valid(model, x_valid, y_valid, criterion):
    """
    Performs forward pass and computes the loss.
    (Applied to validation dataset)

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        x_valid : torch.Tensor
            The minibatch of validation datapoints
        y_valid: torch.Tensor
            The minibatch of validation labels
        criterion : torch.nn.Module
            The loss criterion
    """

    # set model to validation mode
    model.eval()
    # forward pass
    y_pred = model(x_valid)
    # compute loss
    loss = criterion(y_pred, y_valid).item() # .item() gets only the scalar

    return y_pred, loss

@no_grad()
def valid_check(
        model,
        device,
        criterion,
        epoch,
        epoch_ite_valid,
        size_batch,
        writer,
        list_loss_train,
        list_acc_global_train,
        log_dir_figures_valid,
        log_dir_figures_train,
        dict_batches_train,
        list_labels_train,
        list_preds_train,
        log_dir_best_model,
        best_avg_acc_global_valid,
        dict_figures_train,
        logger,
        num_classes,
        debug
):
    """
    Performs a validation pass to monitor and log average losses and
    accuracies, as well as precision-recall curves and confusion
    matrices on the validation set

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        device : torch.device
            The The device that runs the code
        criterion : torch.nn.Module
            The loss criterion
        epoch : int
            The index of the current epoch
        epoch_ite_valid: iter
            The data generator that yields the validation minibatches
        size_batch : int
            The size of each validation minibatch
        writer : torch.utils.tensorboard.SummaryWriter
            The monitoring tool that logs results during the process
        list_loss_train : list
            The list of training losses on the current minibatch
        list_acc_global_train : list
            The list of minibatch-wise average accuracies
        log_dir_figures_valid : str
            The path to the directory where validation figures are saved
        log_dir_figures_train : str
            The path to the directory where training figures are saved
        dict_batches_train : dict
            The dict containing batches of labels categorized by language
        list_labels_train : list
            The list of labels of the training dataset
        list_preds_train : list
            The list of predictions of the training dataset
        log_dir_best_model : str
            The path to the directory where the best model so far is saved
        best_avg_acc_global_valid : float
            The best average validation accuracy so far
        dict_figures_train : dict
            The dict where training performance figures are stored
        logger : stacklogging.Logger
            The monitoring tool that allows to follow results during training
        num_classes : int
            The number of categories to which labels can belong
        debug : int
            The option to run the method in debug mode
    """

    logger.info('\n\n- Validating training up until epoch {}...'.format(epoch))
    # compute average train loss and accuracy
    avg_loss_train = np.mean(list_loss_train)
    avg_acc_global_train = np.mean(list_acc_global_train)
    # compute average validation loss and accuracy
    avg_loss_valid, avg_acc_global_valid = loop_valid(
        model=model,
        criterion=criterion,
        writer=writer,
        device=device,
        epoch_ite_valid=epoch_ite_valid,
        size_batch=size_batch,
        num_classes=num_classes,
        log_dir_figures_valid=log_dir_figures_valid,
        global_step=epoch,
        logger=logger
    )
    # log losses and accuracies to the summary writer
    writer.add_scalars('scalars/loss', {'loss_valid': avg_loss_valid,
                                        'loss_train': avg_loss_train},
                           global_step=epoch)
    writer.add_scalars('scalars/acc', {'acc_global_valid': avg_acc_global_valid,
                                       'acc_global_train': avg_acc_global_train},
                           global_step=epoch)
    # convert labels to one-hot
    for lang, outputs in dict_batches_train.items():
        labels = torch.from_numpy(np.array([[lang2idx[lang]]]*len(outputs)))
        labels_one_hot = torch.nn.functional.one_hot(
            labels,
            num_classes=num_classes
        ).transpose(1, 2)
        outputs = torch.stack(outputs)
        # log precision-recall curves
        writer.add_pr_curve(
            'pr_curves_train/pr_{}'.format(lang),
            labels_one_hot,
            outputs,
            global_step=epoch)
    # plot and log confusion matrix figures
    if list_labels_train and list_preds_train:
        dict_figures_train[fig_titles['cm']] = save_confusion_matrix(
            log_dir=log_dir_figures_train,
            y_true=list_labels_train,
            y_pred=list_preds_train,
            dict_classes=idx2lang,
            save=False
        )
        # add list of figures to summary
        for fig_name, fig_obj in dict_figures_train.items():
            writer.add_figure(
                tag='figures_train/{}'.format(fig_name), figure=fig_obj,
                global_step=epoch, close=True)
            plt.close(fig_obj)
    else:
        logger.info(
            'Either list_labels_train OR list_preds_train or both are empty'
        )
    # delete previous and save new best model checkpoint
    if avg_acc_global_valid > best_avg_acc_global_valid:
        best_avg_acc_global_valid = avg_acc_global_valid
        pattern = '/*' if debug else '*'
        for f in file_io.get_matching_files(log_dir_best_model + pattern):
            os.remove(f)
        best_model_name = 'epoch{:03}-batch{:07}_vacc{:0.5}vloss{:0.5}.pt'.format(
            epoch + 1,
            epoch,
            avg_acc_global_valid,
            avg_loss_valid)
        torch.save(
            model.state_dict(), os.path.join(log_dir_best_model, best_model_name)
        )

    # return performance metric results
    return (
        avg_loss_train,
        avg_acc_global_train,
        avg_loss_valid,
        avg_acc_global_valid,
        best_avg_acc_global_valid
    )


@no_grad()
def loop_valid(
        model,
        device,
        criterion,
        epoch_ite_valid,
        size_batch,
        num_classes,
        log_dir_figures_valid,
        writer,
        global_step,
        logger
):
    """
    Runs the validation forward pass.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        device : torch.device
            The The device that runs the code
        criterion : torch.nn.Module
            The loss criterion
        epoch_ite_valid: iter
            The data generator that yields the validation minibatches
        size_batch : int
            The size of each validation minibatch
        num_classes : int
            The number of categories to which labels can belong
        log_dir_figures_valid : str
            The path to the directory where validation figures are saved
        writer : torch.utils.tensorboard.SummaryWriter
            The monitoring tool that logs results during the process
        global_step : int
            The writer monitoring global index
        logger : stacklogging.Logger
            The monitoring tool that allows to follow results during training
    """
    dict_figures_valid = {}
    list_loss_valid, list_acc_global_valid = [], []
    list_labels_valid, list_preds_valid = [], []
    dict_batches_valid = defaultdict(lambda: [])

    logger.info('\nValidation starts\n')
    for batch_idx_valid, data_valid in enumerate(epoch_ite_valid):
        # obtain validation input and labels from data feeder
        inputs_valid = torch.from_numpy(data_valid[0]).to(device)
        labels_valid = torch.from_numpy(data_valid[1]).long().to(device)
        # save validation labels
        list_labels_valid += list(
            labels_valid.cpu().detach().numpy().astype(int).reshape(size_batch)
        )
        # perform forward pass and compute validation loss
        outputs_valid, loss_valid = step_valid(
            model,
            inputs_valid,
            labels_valid,
            criterion
        )
        # save validation predictions
        list_preds_valid += list(
            np.argmax(outputs_valid.cpu().detach().numpy(),
                      axis=1
                      ).astype(int).reshape(size_batch))
        # save validation losses and accuracies
        list_loss_valid.append(loss_valid)
        acc_global_valid = compute_accuracy(outputs_valid, labels_valid)[-1]
        list_acc_global_valid.append(acc_global_valid)
        # save validation outputs categorized by label
        for label, output in zip(labels_valid, outputs_valid):
            label_key = idx2lang[label.cpu().detach().numpy().astype(int)[0]]
            dict_batches_valid[label_key] += [output]
        # every once in a while monitor validation loop
        if ((batch_idx_valid + 1) % 100) == 0:
            logger.info('-- Validation mini-batch {} of {} processed'.format(
                batch_idx_valid + 1, size_batch), end='\r')
    # convert labels to one-hot
    for lang, outputs in dict_batches_valid.items():
        labels = torch.from_numpy(np.array([[lang2idx[lang]]]*len(outputs)))
        labels_one_hot = torch.nn.functional.one_hot(
            labels, num_classes=num_classes
        ).transpose(1, 2)
        outputs = torch.stack(outputs)
        # log precision-recall curves
        writer.add_pr_curve(
            'pr_curves_valid/pr_{}'.format(lang),
            labels_one_hot,
            outputs,
            global_step=global_step)

    # plot and log confusion matrix
    if list_labels_valid and list_preds_valid:
        dict_figures_valid[fig_titles['cm']] = save_confusion_matrix(
            log_dir=log_dir_figures_valid,
            y_true=list_labels_valid,
            y_pred=list_preds_valid,
            dict_classes=idx2lang,
            save=False)
        # add list of figures to summary
        for fig_name, fig_obj in list(dict_figures_valid.items()):
            writer.add_figure(
                tag='figures_valid/{}'.format(fig_name), figure=fig_obj,
                global_step=global_step, close=True)
            plt.close(fig_obj)

    # return average validation losses and accuracies
    return np.mean(list_loss_valid), np.mean(list_acc_global_valid)


def get_dataset(path_dataset, size_dataset):
    """
    Collects the dataset files.

        Parameters
        ----------
        path_dataset : str
            The path to the full dataset folder
        size_dataset : int
            The amount of datapoints to collect
    """
    list_data_filepaths = []
    for _ in os.listdir(path_dataset):
        list_data_filepaths += [f for f in \
            file_io.get_matching_files('{}/*'.format(path_dataset))[:size_dataset]]


def main(args):
    dict_config = args.__dict__

    # list of adjustable config params
    list_hyper = [
        'size_batch',
        'len_seq',
        'num_layers',
        'num_hidden',
        'size_kernel',
        'learning_rate',
        'drop_out'
    ]

    # create job id from config params to easily identify the results
    hyperparam_id = dict2str(
        {
            key: value for (key, value) in dict_config.items() \
            if key in list_hyper
        }
    )[2:]

    # create directiry for logging
    log_dir = os.path.join('logs_', hyperparam_id)

    # save config to JSON file
    with file_io.FileIO(os.path.join(log_dir, 'config.json'), mode='w+') as fp:
        json.dump(dict_config, fp)

    # create summary writer object
    log_dir_writer = log_dir if args.debug else '/tmp/writer/'
    writer = SummaryWriter(log_dir=log_dir_writer)
    logger = stacklogging.getLogger()
    logger.setLevel(logging.INFO)

    # set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # check for CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    device = torch.device('cuda:0') \
        if not args.debug and torch.cuda.is_available() \
        else 'cpu'

    # create logging folders
    log_dir_figures_train = os.path.join(log_dir, 'figures/train')
    if not os.path.isdir(log_dir_figures_train):
        os.makedirs(log_dir_figures_train)
    log_dir_figures_valid = os.path.join(log_dir, 'figures/valid')
    if not os.path.isdir(log_dir_figures_valid):
        os.makedirs(log_dir_figures_valid)
    log_dir_best_model = os.path.join(log_dir, 'best_model')
    if not os.path.isdir(log_dir_best_model):
        os.makedirs(log_dir_best_model)

    logger.info('\nStarting training process...')

    # set the model architecture
    list_conv_depths = [args.num_hidden for n in range(args.num_layers)]

    # construct model
    model = TCN_SLID(
        size_in=args.len_seq,
        size_out=args.num_classes,
        list_conv_depths=list_conv_depths,
        size_kernel=args.size_kernel,
        stride=args.size_stride,
        dropout=args.drop_out)

    # move model to GPU if present
    if device is not 'cpu':
        model.to(device)
    logger.info('\nModel compiled.')

    logger.info('\nSetting up optimizers and losses... ', end='')
    # set optimizer and loss criterion
    optimizer = getattr(optim, 'RMSprop')(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    logger.info('\nOptimizer and loss set.')

    # create data indices for training and validation splits
    list_filepaths_train,\
    list_filepaths_test,\
    list_filepaths_valid = split_train_test_valid(get_dataset(
        path_dataset=args.path_dataset,
        size_dataset=args.size_dataset
    ))

    # Initializing data feeder
    epoch_ite_train, epoch_ite_test, epoch_ite_valid = data_generator_training(
        filenames_train=list_filepaths_train[:] if args.debug else list_filepaths_train,
        filenames_test=list_filepaths_test[:1] if args.debug else list_filepaths_test,
        filenames_valid=list_filepaths_valid[:] if args.debug else list_filepaths_valid,
        len_chunk=args.len_seq,
        size_batch=args.size_batch,
        logger=logger,
        sr=args.sr
    )

    # materialize data feeders
    epoch_ite_train = list(epoch_ite_train())
    shuffle_batches_train = True
    if shuffle_batches_train:
        np.random.shuffle(epoch_ite_train)
    epoch_ite_valid = list(epoch_ite_valid())
    shuffle_batches_valid = True
    if shuffle_batches_valid:
        np.random.shuffle(epoch_ite_valid)

    # epoch loop starts
    for epoch in range(args.epochs):
        # declare epoch-loop variables
        list_loss_train, list_acc_global_train = [], []
        list_labels_train, list_preds_train = [], []
        dict_figures_train = {}
        dict_batches_train = defaultdict(lambda: [])
        best_avg_acc_global_valid = -1
        time_start = time.time()

        # training loop starts
        for batch_idx_train, data_train in enumerate(epoch_ite_train):
            inputs_train = torch.from_numpy(data_train[0]).to(device)
            labels_train = torch.from_numpy(data_train[1]).long().to(device)
            list_labels_train += list(
                labels_train.cpu().detach().numpy().astype(int).reshape(args.size_batch)
            )
            # forward step
            outputs_train, loss_train = step_train(
                model=model,
                x=inputs_train,
                y=labels_train,
                criterion=criterion,
                optimizer=optimizer
            )
            # save predictions
            list_preds_train += list(np.argmax(outputs_train.cpu().detach().numpy(), axis=1)
                                    .astype(int).reshape(args.size_batch))
            # save losses
            list_loss_train.append(loss_train.cpu().detach().numpy())
            # save average accuracies
            acc_global_train = compute_accuracy(outputs_train, labels_train)[-1]
            list_acc_global_train.append(acc_global_train)
            # create dict of outputs for facilitating validation loop
            for label, output in zip(labels_train, outputs_train):
                label_key = idx2lang[label.cpu().detach().numpy().astype(int)[0]]
                dict_batches_train[label_key] += [output]
            # end of training loop

        # every certain number of epochs, report performance on the validation set
        if (epoch + 1) % args.epochs_per_check == 0:
            avg_loss_train, \
            avg_acc_global_train, \
            avg_loss_valid, \
            avg_acc_global_valid, \
            best_avg_acc_global_valid = valid_check(
                model=model,
                device=device,
                criterion=criterion,
                epoch=epoch + 1,
                epoch_ite_valid=epoch_ite_valid,
                size_batch=args.size_batch,
                writer=writer,
                list_loss_train=list_loss_train,
                list_acc_global_train=list_acc_global_train,
                log_dir_figures_valid=log_dir_figures_valid,
                log_dir_figures_train=log_dir_figures_train,
                dict_batches_train=dict_batches_train,
                list_labels_train=list_labels_train,
                list_preds_train=list_preds_train,
                log_dir_best_model=log_dir_best_model,
                best_avg_acc_global_valid=best_avg_acc_global_valid,
                dict_figures_train=dict_figures_train,
                num_classes=args.num_classes,
                logger=logger,
                debug=args.debug
            )
            time_end_train = time.time()
            duration_train = time_end_train - time_start

            # report training and validation losses on epoch end
            epoch_string = training_output_string.format(
                ep=epoch + 1,
                loss_train=avg_loss_train,
                acc_train=avg_acc_global_train,
                loss_valid=avg_loss_valid,
                acc_valid=avg_acc_global_valid,
                t_train=duration_train)
            logger.info(epoch_string)

    # end of epoch loop
    logger.info('\nTraining done.')
    logger.info('That\'s all folks!')

    # close SummaryWriter object
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size-dataset', type=int, default=77,
                        help='# of files used as dataset (default: 77)')
    parser.add_argument('--sr', type=int, default=44100,
                        help='Original input sampling rate')
    parser.add_argument('--debug', type=bool, default=False,
                        help='run in debug (local) mode (default: False)')
    parser.add_argument('--input-file-format', type=str, default='wav',
                        help='input file format (default: WAV)')
    parser.add_argument('--num-classes', type=int, default=7,
                        help='Number of target languages (default: 7)')
    parser.add_argument('--len-seq', type=int, default=4410,
                        help='Input sequence length (default: 4410)')
    parser.add_argument('--size-kernel', type=int, default=3,
                        help='kernel size (default: 3)')
    parser.add_argument('--size-stride', type=int, default=1,
                        help='stride size (default: 1)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='# of layers (default: 4)')
    parser.add_argument('--num-hidden', type=int, default=150,
                        help='# of hidden units per layer (default: 150)')
    parser.add_argument('--size-batch', type=int, default=128,
                        help='training batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit (default: 2)')
    parser.add_argument('--epochs-per-check', type=int, default=5,
                        help='number of epochs before monitoring report (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--drop_out', type=float, default=0.25,
                        help='dropout applied to layers (default: 0.25)')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed (default: 7)')
    parser.add_argument('--data-dir', type=str, default='',
                        help='The data directory',)
    parser.add_argument('--job-dir', type=str, default='',
                        help='The job directory')
    parser.add_argument('--job-id', type=str, default='',
                        help='The job id (timestamp)')

    parse_args, unknown = parser.parse_known_args()

    main(parse_args)
