""""
This script performs the preprocessing steps: Segmentation
and Vocals Isolation. It launches a Spark session and
sends partitions of the input dataset to multiple workers,
each of them applying the preprocessing algorithms to their
corresponding partition.
"""


import time
import argparse
import os
import torch

from pyspark.sql import SparkSession

from utils.data_utils import jsonify
from utils.audio_utils import load_audio_file_final, write_audio_file
from utils.settings import hyper_parameters

from vocals_isolation import isolate_vocals
from segmentation import find_relevant_segments

import stacklogging, logging


def main(args):
    init_time = int(time.time())

    # create logger
    logger = stacklogging.getLogger()
    logger.setLevel(logging.INFO)

    # start pyspark session
    logger.info('\nLaunching Spark Session...')
    application_name = args.application_name
    spark = SparkSession \
        .builder \
        .master('local[]' if args.debug else 'yarn') \
        .appName(application_name) \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # collect list of audio files
    list_filenames = []
    for lang in os.listdir(args.path_dataset):
        dir_lang = os.path.join(args.path_dataset, lang)
        list_filenames += [os.path.join(dir_lang, song) \
                                for song in os.listdir(dir_lang)]

    # read all audio filenames and load them as RDD objects
    rdd_f00bd01 = sc.parallelize(list_filenames, numSlices=80)

    # read and load pre-trained neural network weights
    path_weights_enc = './madtwinnet_weights/rnn_enc/rnn_enc.pt'
    path_weights_dec = './madtwinnet_weights/rnn_dec/rnn_dec.pt'
    path_weights_fnn = './madtwinnet_weights/fnn/fnn.pt'
    path_weights_den = './madtwinnet_weights/denoiser/denoiser.pt'
    logger.info('\nLoading MadTwinNet weights...')
    weights = [
        torch.load(path_weights_enc),
        torch.load(path_weights_dec),
        torch.load(path_weights_fnn),
        torch.load(path_weights_den)]
    logger.info('\nWeights loaded successfully')

    # step 0: load the mix waveforms
    rdd_d00sr01_f1 = rdd_f00bd01.map(
        lambda f00bd01: load_audio_file_final(f00bd01, sr=args.sampling_rate))

    # step 1: isolate the vocals from mix and remove the quiet segments
    rdd_vm00vw01vws02sr03_f1 = rdd_d00sr01_f1.map(
        lambda d00sr01_f1: isolate_vocals(
            d00sr01_f1=d00sr01_f1,
            weights_d0=weights,
            rnn_enc_output_dim=hyper_parameters['rnn_enc_output_dim'],
            original_input_dim=hyper_parameters['original_input_dim'],
            reduced_dim=hyper_parameters['reduced_dim'],
            context_length=args.context_length,
            window_size=hyper_parameters['window_size'],
            fft_size=hyper_parameters['fft_size'],
            hop_size=hyper_parameters['hop_size'],
            seq_length=hyper_parameters['seq_length'],
            batch_size=args.batch_size,
            top_db=hyper_parameters['top_db'],
            ))

    # step 2: find the segments that are relevant for language identification
    rdd_segswav00sr01_f1 = rdd_vm00vw01vws02sr03_f1.map(
        lambda vm00vw01vws02sr03_f1: find_relevant_segments
        (
            (
                (vm00vw01vws02sr03_f1[0][2], vm00vw01vws02sr03_f1[0][3]),
                vm00vw01vws02sr03_f1[1]
            ),
            hyper_parameters['n_segments'],
            hyper_parameters['segment_len_sec'],
            args.size_fft_chroma,
            hyper_parameters['n_ite_autocorrelation']
        )
    )

    # write data to audio format
    rdd_segswav00sr01_f1 = rdd_segswav00sr01_f1.map(
                lambda segswav00sr01_f1: write_audio_file(
                    (
                        segswav00sr01_f1,
                        init_time,
                        os.path.basename(os.path.dirname(segswav00sr01_f1[1]))
                    )
                )
    )

    # save segments into json file
    rdd_json = rdd_segswav00sr01_f1.map(
        lambda segswav00sr01_f1_vws2: jsonify(segswav00sr01_f1_vws2))

    path_outputs_down_text = './outputs'
    rdd_json.saveAsTextFile(path_outputs_down_text)

    sc.stop()

    # report time taken
    total_time = time.time() - init_time
    logger.info('Total time: {t:6.2f} sec(s)'.format(t=total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False,
                        help='run in debug (local) mode (default: False)')
    parser.add_argument('--path-dataset', type=str, default='/tmp/',
                        help='The path to the dataset to preprocess')
    parser.add_argument('--application_name', type=str, default='APP NAME',
                        help='The name of the Spark session')
    parser.add_argument('--sampling-rate', type=int, default=44100,
                        help='Sampling rate')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--size-fft-chroma', type=int, default=1024,
                        help='Size of the FFT window required for computing the chroma')
    parser.add_argument('--context-length', type=int, default=10,
                        help='Context length')

    parse_args, unknown = parser.parse_known_args()

    main(parse_args)