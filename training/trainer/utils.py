""""
This script contains helper variables and methods used for training NNs
for Singing Language Identification.
"""


import os
import numpy as np
import torch


# training monitoring strings
print_losses_string = 'batch {:07} || TRAIN LOSS:{} || VALID LOSS:{}\n'
training_output_string = 'Epoch: {ep:03} << losses >> ' \
                         'train:{loss_train:6.4f} | valid:{loss_valid:6.4f} ' \
                         '|| << accuracies >> ' \
                         'train:{acc_train:6.4f} | valid.:{acc_valid:6.4f} ' \
                         '|| << times >> train: {t_train:6.2f} sec(s)'

# language identifiers
lang2idx = {
    'ja': 0,
    'de': 1,
    'es': 2,
    'fr': 3,
    'en': 4,
    'it': 5,
    'pt': 6
}

idx2lang = {
    0: 'ja',
    1: 'de',
    2: 'es',
    3: 'fr',
    4: 'en',
    5: 'it',
    6: 'pt'
}


def split_train_test_valid(list_dataset, split_valid=.2, split_test=.2, shuffle=True):
    """
    Splits the full dataset into train, validation, and test subsets.

        Parameters
        ----------
        list_dataset : list
            The list of all files in the full dataset
        split_valid : float
            The ratio of datapoints that go into the validation set
        split_test: float
            The ratio of datapoints that go into the test set
        shuffle : bool
            The option to shuffle training datapoints
        optimizer: torch.optim
            The optimization algorithm
    """

    len_dataset = len(list_dataset)
    indices_train_test = list(range(len_dataset))
    split_train_test = int(np.floor(split_test * len_dataset))
    if shuffle:  # we shuffle the training files
        np.random.shuffle(indices_train_test)
    indices_train_valid = indices_train_test[split_train_test:]
    indices_test = indices_train_test[:split_train_test]
    list_dataset_test = [list_dataset[i] for i in indices_test]
    split_train_valid = int(np.floor(split_valid * len(indices_train_valid)))
    indices_train = indices_train_valid[split_train_valid:]
    indices_valid = indices_train_valid[:split_train_valid]
    list_dataset_train = [list_dataset[i] for i in indices_train]
    list_dataset_valid = [list_dataset[i] for i in indices_valid]

    return list_dataset_train, list_dataset_test, list_dataset_valid


def compute_accuracy(y_pred, y_true):
    """
    Computes the average accuracy given predictions and true labels

        Parameters
        ----------
        y_true : torch.Tensor
            The list of true labels
        y_pred : torch.Tensor
            The list of predicted labels
    """
    confidences, winners = torch.squeeze(y_pred).max(dim=1)
    corrects = (winners == torch.squeeze(y_true))
    accuracy = (corrects.sum().float() / float(y_true.size(0))).cpu().detach().numpy()

    return confidences, winners, corrects, accuracy


def copy_file_to_gcs(path_source, path_dest):
    """
    Reads a local file and writes it in a Google Cloud Storage bucket.

        Parameters
        ----------
        path_source : str
            The path of the local file
        path_dest : str
            The path of the destination file
    """
    from tensorflow.python.lib.io import file_io

    with file_io.FileIO(path_source, mode='rb') as input_f:
        with file_io.FileIO(path_dest, mode='w+') as output_f:
            output_f.write(input_f.read())


def audio_load(f):
    """
    Loads an audio from a file and converts it into a numpy array.

        Parameters
        ----------
        f : str
            The audio file to load
    """
    import soundfile as sf

    try:
        data = sf.read(f, dtype='float32')
    except Exception as e:
        print('Exception caught:', e)
        print('File responsible:', f)
        data = None, None

    return data


def data_generator_training(
        filenames_train,
        filenames_test,
        filenames_valid,
        len_chunk,
        size_batch,
        logger,
        sr,
        len_seg=5
):
    """
    Creates iterator objects for training, testing, and validation datasets.

        Parameters
        ----------
        filenames_train : list
            The list of training files
        filenames_test : list
            The list of testing files
        filenames_valid : list
            The list of validation files
        len_chunk : int
            The length of the datapoint fed to the NN classifier
        size_batch : int
            The size of the batch of datapoints
        logger: logging.Logger
            The object that helps monitor the training process
        sr : int
            The sampling rate used when loading an audio file
        len_seg : int (optional)
            The correct length of each segment
    """
    import numpy as np

    # training data epoch-wise iter() generator
    def training_epoch_ite():
        # create accumulator variables
        segment_samples_acc, segment_labels_acc = [], []
        # iterate over list of files
        for file_idx, filename in enumerate(filenames_train):
            # extract language identifier
            segment_lang = os.path.basename(os.path.dirname(filename))
            # if filename is present, load audio
            if (type(filename) == str) and (len(filename) > 0):
                segment_samples, _ = audio_load(filename, None)
                # if audio is loaded correctly, reshape to adapt to neural network input
                if any(segment_samples) and segment_samples.shape[0] == len_seg*sr:
                    segment_samples = segment_samples.reshape(-1, len_chunk)
                    segment_labels = np.empty((segment_samples.shape[0], 1))
                    segment_labels [:,:] = lang2idx[segment_lang]
                    segment_samples_acc.append(segment_samples)
                    segment_labels_acc.append(segment_labels)
                    logger.info('-- Processing training data file {} ...'.format(file_idx))
            # once we have loaded enough inputs, we build a batch
            if (len(segment_samples_acc) * segment_samples.shape[0]) == size_batch:
                segment_samples = np.vstack(segment_samples_acc)
                segment_labels = np.vstack(segment_labels_acc)
                for batch in range(segment_samples.shape[0] // size_batch):
                    b_start = batch * size_batch
                    b_end = (batch + 1) * size_batch
                    segment_samples_batch = segment_samples[b_start:b_end, :]
                    segment_samples_batch = np.expand_dims(
                        segment_samples_batch,
                        axis=1
                    ).astype(np.float32)
                    segment_labels_batch = segment_labels[b_start:b_end, :]

                    yield segment_samples_batch, segment_labels_batch
                # empty accumulators
                segment_samples_acc, segment_labels_acc = [], []

    # testing data epoch-wise iter() generator
    def test_epoch_ite():
        # create accumulator variables
        segment_samples_acc, segment_labels_acc = [], []
        # iterate over list of files
        for file_idx, filename in enumerate(filenames_test):
            # extract language identifier
            segment_lang = os.path.basename(os.path.dirname(filename))
            # if filename is present, load audio
            if (type(filename) == str) and (len(filename) > 0):
                segment_samples, _ = audio_load(filename, None)
                # if audio is loaded correctly, reshape to adapt to the NN input
                if any(segment_samples) and segment_samples.shape[0] == len_seg*sr:
                    segment_samples = segment_samples.reshape(-1, len_chunk)
                    segment_labels = np.empty((segment_samples.shape[0], 1))
                    segment_labels [:,:] = lang2idx[segment_lang]
                    segment_samples_acc.append(segment_samples)
                    segment_labels_acc.append(segment_labels)
                    logger.info('-- Processing test data file {} ...'.format(file_idx))
            # once we have loaded enough inputs, we build a batch
            if (len(segment_samples_acc) * segment_samples.shape[0]) == size_batch:
                segment_samples = np.vstack(segment_samples_acc)
                segment_labels = np.vstack(segment_labels_acc)
                for batch in range(segment_samples.shape[0] // size_batch):
                    b_start = batch * size_batch
                    b_end = (batch + 1) * size_batch
                    segment_samples_batch = segment_samples[b_start:b_end, :]
                    segment_samples_batch = np.expand_dims(
                        segment_samples_batch,
                        axis=1
                    ).astype(np.float32)
                    segment_labels_batch = segment_labels[b_start:b_end, :]

                    yield segment_samples_batch, segment_labels_batch
                # empty accumulators
                segment_samples_acc, segment_labels_acc = [], []

    # validation data epoch-wise iter() generator
    def validation_epoch_ite():
        # create accumulator variables
        segment_samples_acc, segment_labels_acc = [], []
        # iterate over list of files
        for file_idx, filename in enumerate(filenames_valid):
            # extract language identifier
            segment_lang = os.path.basename(os.path.dirname(filename))
            # if filename is present, load audio
            if (type(filename) == str) and (len(filename) > 0):
                segment_samples, _ = audio_load(filename, None)
                # if audio is loaded correctly, reshape to adapt to neural network input
                if any(segment_samples) and segment_samples.shape[0] == len_seg*sr:
                    segment_samples = segment_samples.reshape(-1, len_chunk)
                    segment_labels = np.empty((segment_samples.shape[0], 1))
                    segment_labels [:,:] = lang2idx[segment_lang]
                    segment_samples_acc.append(segment_samples)
                    segment_labels_acc.append(segment_labels)
                    logger.info('-- Processing validation data file {} ...'.format(file_idx))
            # once we have loaded enough inputs, we build a batch
            if (len(segment_samples_acc) * segment_samples.shape[0]) == size_batch:
                segment_samples = np.vstack(segment_samples_acc)
                segment_labels = np.vstack(segment_labels_acc)
                for batch in range(segment_samples.shape[0] // size_batch):
                    b_start = batch * size_batch
                    b_end = (batch + 1) * size_batch
                    segment_samples_batch = segment_samples[b_start:b_end, :]
                    segment_samples_batch = np.expand_dims(
                        segment_samples_batch,
                        axis=1
                    ).astype(np.float32)
                    segment_labels_batch = segment_labels[b_start:b_end, :]
                    
                    yield segment_samples_batch, segment_labels_batch
                # empty accumulators
                segment_samples_acc, segment_labels_acc = [], []

    return training_epoch_ite, test_epoch_ite, validation_epoch_ite



# maps a dict into a single string
def dict2str(d):
    s = '_'
    for k,v in d.items():
        terms = k.split('_')
        if len(terms) == 1:
            s += '_' + terms[0] + str(v)
            continue
        s += '_' + ''.join([t[0] for t in terms]) + str(v)
    return s