""""
This script contains some helper functions needed for
the preprocessing of the dataset. Some of the functions have
been refactored from https://github.com/dr-costas/mad-twinnet.
"""


# dumps a file path in json format
def jsonify(path):
    import json

    return json.dumps({'file_path': p[1]})


# since PySpark requires partitions, we make an iter() wrapper
def P_jsonify(P_path):

    P_out = []
    for path in P_path:
        P_out.append(jsonify(path))

    return iter(P_out)


# maps a dict to a single string
def dict2str(d):
    s = '_'
    for k,v in d.items():
        terms = k.split('_')
        if len(terms) == 1:
            s += '_' + terms[0] + str(v)
            continue
        s += '_' + ''.join([t[0] for t in terms]) + str(v)
    return s


# loads neural network weights as torch tensors
def load_weights(fn_data, path_download='/tmp/'):
    import torch
    import os

    path_weights_file = fn_data[0]
    binary_weights_data = fn_data[1]

    download_path = path_download + os.path.basename(path_weights_file)
    with open(download_path, 'wb+') as f:
        f.write(binary_weights_data)

    madtwinnet_module_dict = torch.load(download_path)
    os.remove(download_path)

    return madtwinnet_module_dict


"""
The rest of the methods below have been refactored from https://github.com/dr-costas/mad-twinnet
"""



def data_generator_testing(d00sr01_f1, window_size, fft_size, hop_size, seq_length, context_length,
                           batch_size):
    from scipy import signal
    hamming_window = signal.hamming(window_size, True)

    testing_ite = _get_data_testing(d00sr01_f1=d00sr01_f1, window_values=hamming_window, fft_size=fft_size,
                                       hop_size=hop_size, seq_length=seq_length, context_length=context_length,
                                       batch_size=batch_size)

    return testing_ite


def _get_data_testing(d00sr01_f1, window_values, fft_size, hop_size,
                      seq_length, context_length, batch_size):
    from utils.audio_utils import stft

    mix = d00sr01_f1[0][0]
    mix_magnitude, mix_phase = stft(mix, window_values, fft_size, hop_size)
    # data reshaping (magnitude and phase)
    mix_magnitude, mix_phase, _ = _make_overlap_sequences(mixture=mix_magnitude, voice=mix_phase, bg=None,
                                                             l_size=seq_length, o_lap=context_length * 2,
                                                             b_size=batch_size)
    return mix, mix_magnitude, mix_phase


def _make_overlap_sequences(mixture, voice, bg, l_size, o_lap, b_size):
    import numpy as np

    trim_frame = mixture.shape[0] % (l_size - o_lap)
    trim_frame -= (l_size - o_lap)
    trim_frame = np.abs(trim_frame)

    if trim_frame != 0:
        mixture = np.pad(mixture, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        voice = np.pad(voice, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        # bg = np.pad(bg, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

    mixture = np.lib.stride_tricks.as_strided(
        mixture,
        shape=(int(mixture.shape[0] / (l_size - o_lap)), l_size, mixture.shape[1]),
        strides=(mixture.strides[0] * (l_size - o_lap), mixture.strides[0], mixture.strides[1])
    )
    mixture = mixture[:-1, :, :]

    voice = np.lib.stride_tricks.as_strided(
        voice,
        shape=(int(voice.shape[0] / (l_size - o_lap)), l_size, voice.shape[1]),
        strides=(voice.strides[0] * (l_size - o_lap), voice.strides[0], voice.strides[1])
    )
    voice = voice[:-1, :, :]

    while mixture.shape[0] < b_size:
        b_size //= 2
    if b_size > 0:
        b_trim_frame = (mixture.shape[0] % b_size)
        if b_trim_frame != 0:
            mixture = mixture[:-b_trim_frame, :, :]
            voice = voice[:-b_trim_frame, :, :]

    return mixture, voice, None



def data_process_results_testing(voice_predicted, window_size, mix_magnitude, mix_phase, hop, context_length,
                                    top_db):
    from utils.audio_utils import i_stft, split_audio

    voice_predicted.shape = (voice_predicted.shape[0] * voice_predicted.shape[1], window_size)
    mix_magnitude, mix_phase = _context_based_reshaping(mix_magnitude, mix_phase, context_length, window_size)

    voice_hat = i_stft(voice_predicted, mix_phase, window_size, hop)
    voice_hat_split = split_audio(voice_hat, top_db=top_db)

    return None, voice_hat_split


def _context_based_reshaping(mix, voice, context_length, window_size):
    import numpy as np

    mix = np.ascontiguousarray(mix[:, context_length:-context_length, :], dtype=np.float32)
    mix.shape = (mix.shape[0] * mix.shape[1], window_size)
    voice = np.ascontiguousarray(voice[:, context_length:-context_length, :], dtype=np.float32)
    voice.shape = (voice.shape[0] * voice.shape[1], window_size)

    return [], voice