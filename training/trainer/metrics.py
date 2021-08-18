""""
This script contains a set metrics used for evaluating the performance
of the models trained for the task of Singing Language Identification.
"""


import os
from itertools import product, cycle

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, precision_recall_curve, \
                                roc_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from .utils import idx2lang, copy_file_to_gcs


colorCycle = cycle([
    'aqua', 'xkcd:azure', 'beige', 'black', 'blue',
    'chartreuse', 'chocolate', 'coral', 'xkcd:crimson', 'grey', 'darkblue',
    'xkcd:fuchsia', 'gold', 'indigo', 'khaki', 'lightgreen', 'lightblue', 'lavender',
    'olive', 'red', 'pink', 'orchid', 'plum', 'purple',
    'tomato', 'teal', 'violet', 'wheat', 'yellow'
])


fig_titles = {
    'cm': 'cm.png',
    'multipr': 'multipr.png',
    'iso-f1': 'iso-f1.png'
}


def compute_confusion_matrix(y_true, y_pred, classes):
    """
    Computes the confusion matrix to evaluate the performance of a classifier.

        Parameters
        ----------
        y_true : list
            The list of true labels
        y_pred : list
            The list of predicted labels
        classes: list
            The list of categories to which the labels can belong to
    """
    y_pred = np.around(
        np.clip(
            y_pred,
            a_min=np.min(classes),
            a_max=np.max(classes)
        )
    )
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    num_labels_vectorized = cm.sum(axis=1)[:, np.newaxis]
    cm = np.divide(cm.astype('float'),
                   num_labels_vectorized)
    return cm

def plot_confusion_matrix(y_true, y_pred, dict_classes, n_classes):
    """
    Plots the confusion matrix.

        Parameters
        ----------
        y_true : list
            The list of true labels
        y_pred : list
            The list of predicted labels
        dict_classes: dict
            The dict of categories to which the labels can belong to
        n_classes: int
            The number of possible categories
    """
    classes = list(range(n_classes))
    fig, ax = plt.subplots()
    cm = compute_confusion_matrix(y_true, y_pred, classes)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix')
    fig.colorbar(im, ax=ax)
    # label the x axis ticks
    ax.set_xticks(classes)
    ax.set_xticklabels([dict_classes[c] for c in classes])
    xticklabels = ax.get_xticklabels()
    # rotate x axis ticks
    for xtlabel in xticklabels:
        xtlabel.set_rotation(45)
    # label the y axis ticks
    ax.set_yticks(classes)
    ax.set_yticklabels([dict_classes[c] for c in classes])
    # set the format of the cell values
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig

def save_confusion_matrix(
        log_dir,
        y_true,
        y_pred,
        dict_classes,
        n_classes=len(idx2lang),
        tmp_dir='/tmp/figures',
        save=True
):
    """
    Saves the confusion matrix as an image.

        Parameters
        ----------
        log_dir : str
            The logging directory in the cloud
        y_true : list
            The list of true labels
        y_pred : list
            The list of predicted labels
        dict_classes: dict
            The dict of categories to which the labels can belong to
        n_classes: int
            The number of possible categories
        tmp_dir : str
            The local temporary directory where to store the image
        save : bool
            The option to just plot or plot and save the figure

    """
    # plot and return confusion matrix figure
    fig = plot_confusion_matrix(y_true, y_pred, dict_classes, n_classes)
    # save figure
    if save:
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        fig.savefig(
            os.path.join(tmp_dir, fig_titles['cm']),
            dpi=600,
            bbox_inches='tight'
        )
        copy_file_to_gcs(
            path_source=os.path.join(tmp_dir, fig_titles['cm']),
            path_dest=os.path.join(log_dir, fig_titles['cm']))

    return fig



def multipr(y_true, y_pred, n_classes):
    """
    Computes precision-recall curves.

        Parameters
        ----------
        y_true : list
            The list of true labels
        y_pred : list
            The list of predicted labels
        n_classes: int
            The number of possible categories
    """
    # convert labels to one-hot vectors
    y_true = label_binarize(y_true, classes=range(n_classes))
    y_pred = np.array(y_pred)

    precision = dict()
    recall = dict()
    average_precision = dict()
    # compute precision-recall values
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(
            y_true[:, i],
            y_pred[:, i]
        )

    # micro-average quantifies score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true.ravel(),
        y_pred.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true,
        y_pred,
        average="micro"
    )
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    return precision, recall, average_precision


def save_multipr(
        log_dir,
        precision,
        recall,
        average_precision,
        tmp_dir,
        save=True
):
    """
    Saves precision-recall curves as images.

        Parameters
        ----------
        log_dir : str
            The logging directory in the cloud
        precision : dict
            The dict of precision values
        recall : dict
            The dict of recall values
        average_precision: dict
            The dict of averaged precision values
        tmp_dir : str
            The local temporary directory where to store the image
        save : bool
            The option to just plot or plot and save the figure
    """

    # create figure
    fig, ax = plt.subplots()
    step_kwargs = {'step': 'post'}
    ax.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    ax.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                     **step_kwargs)

    # set figure title, axes labels and limits
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    # save figure
    if save:
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        fig.savefig(
            os.path.join(tmp_dir, fig_titles['multipr']),
            dpi=600,
            bbox_inches='tight'
        )
        copy_file_to_gcs(
            path_source=os.path.join(tmp_dir, fig_titles['multipr']),
            path_dest=os.path.join(log_dir, fig_titles['multipr']))

    return fig


def save_isof1(
        log_dir,
        precision,
        recall,
        average_precision,
        n_classes,
        tmp_dir,
        save=True
):
    """
    Computes and saves ISO-F1 curves as images.

        Parameters
        ----------
        log_dir : str
            The logging directory in the cloud
        precision : dict
            The dict of precision values
        recall : dict
            The dict of recall values
        average_precision: dict
            The dict of averaged precision values
        n_classes : int
            The number of possible categories
        tmp_dir : str
            The local temporary directory where to store the image
        save : bool
            The option to just plot or plot and save the figure
    """

    # create figure
    fig, ax = plt.subplots(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = ax.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colorCycle):
        l, = ax.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(idx2lang[i], average_precision[i]))

    fig.subplots_adjust(bottom=0.25)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Extension of Precision-Recall curve to multi-class')
    ax.legend(lines, labels, loc=(0, -.38), prop=dict(size=10))

    # save figure
    if save:
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        fig.savefig(
            os.path.join(tmp_dir, fig_titles['iso-f1']),
            dpi=600,
            bbox_inches='tight'
        )
        copy_file_to_gcs(
            path_source=os.path.join(tmp_dir, fig_titles['iso-f1']),
            path_dest=os.path.join(log_dir, fig_titles['iso-f1']))
    
    return fig

