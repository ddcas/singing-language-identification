from collections import OrderedDict

"""The configuration parameters for the preprocessing steps
"""

# run in debug mode?
debug = False
_debug_suffix = '_debug' if debug else ''

# strings to monitor preprocessing
usage_output_string_per_example = '-- File {f} processed. Time: {t:6.2f} sec(s)'
usage_output_string_total = 'All files processed. Total time: {t:6.2f} sec(s)'

# dict that maps language labels to one-hot vectors
dir2lang = {'ja': [0]*7,
            'de': [0, 1, 0, 0, 0, 0, 0],
            'es': [0, 0, 1, 0, 0, 0, 0],
            'fr': [0, 0, 0, 1, 0, 0, 0],
            'en': [0, 0, 0, 0, 1, 0, 0],
            'it': [0, 0, 0, 0, 0, 1, 0],
            'pt': [0, 0, 0, 0, 0, 0, 1]
            }


# config parameters
hyper_parameters = OrderedDict({
    # audio resolution
    'sr': 22050,
    # vocals isolation step
    'original_input_dim': 2049,
    'reduced_dim': 744,
    'context_length': 10,
    'window_size': 2049,
    'fft_size': 4096,
    'hop_size': 384,
    'seq_length': 60,
    'batch_size': 1,
    'top_db': 30,
    # segmentation step
    'n_segments': 25,
    'segment_len_sec': 5,
    'size_fft_chroma': 1024,
    'n_ite_autocorrelation': 2,
    'cliper_song': 5
})
hyper_parameters.update({
    'rnn_enc_output_dim': 2 * hyper_parameters['reduced_dim']
})

