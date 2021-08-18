""""
This script contains some helper functions needed for
the preprocessing of the dataset. Some of the functions have
been refactored from https://github.com/dr-costas/mad-twinnet.
"""



def load_audio_file(fn_data, sr):
    """
    Loads an audio from a file and converts it into a numpy array.

        Parameters
        ----------
        fn_data : tuple
            The input tuple that contains the audio filename and data
        sr : int
            The sampling rate
    """
    import os

    if type(fn_data) == str:
        path_audio_file = fn_data
        d0sr1 = (ffmpeg_decode(path_audio_file, sr=sr), sr)

    else:
        path_audio_file = fn_data[0]
        binary_audio_data = fn_data[1]
        download_path = 'tmp/' + os.path.basename(path_audio_file)
        with open(download_path, 'wb+') as f:
            f.write(binary_audio_data)

        d0sr1 = (ffmpeg_decode(download_path, sr=sr), sr)
        os.remove(download_path)

    print('-- Song {} loaded'.format(path_audio_file))

    return d0sr1, path_audio_file


# since PySpark requires partitions, we make an iter() wrapper
def P_load_audio_file(P_fn_data, sr):

    P_out = []
    for fn_data in P_fn_data:
        P_out.append(load_audio_file(fn_data[0], sr))

    return iter(P_out)


def write_audio_file(
        segswav00sr01_f1_t2_l3,
        path_outputs,
        path_gcp,
        len_seg,
        audio_format='FLAC'):
    """
    Writes a list of segments into audio files.

        Parameters
        ----------
        segswav00sr01_f1_t2_l3 : tuple
            The input tuple that contains the list of segments
        path_outputs : str
            The local path where to save the output files
        path_cloud : str
            The cloud url where to save the output files
        len_seg : int
            The correct length of each audio segment
        audio_format : str (optional)
            The audio format of the files

    """
    import os
    import soundfile as sf
    from tensorflow.python.lib.io import file_io

    # extract variables from input tuple
    list_segments = segswav00sr01_f1_t2_l3[0][0][0]
    sr = segswav00sr01_f1_t2_l3[0][0][1]
    filename = segswav00sr01_f1_t2_l3[0][1]
    timestamp = segswav00sr01_f1_t2_l3[1]
    lang_id = segswav00sr01_f1_t2_l3[2]
    song_id = os.path.basename(filename)

    if not os.path.isdir(path_outputs):
        os.makedirs(path_outputs)

    num_segments = len(list_segments)
    for i, segment in enumerate(list_segments):
        abs_path = '{}/{}_{}'.format(path_outputs, song_id, i)
        if len(segment) == (len_seg * sr):
            sf.write(
                file=abs_path,
                data=segment,
                samplerate=sr,
                format=audio_format)
            try:
                path_segment_up = os.path.join(path_gcp, lang_id, '{}_{}'.format(song_id, i))
                with file_io.FileIO(abs_path, mode='rb') as input_f:
                    with file_io.FileIO(path_segment_up, mode='w+') as output_f:
                        output_f.write(input_f.read())
                        print('-- Segment {} of {} from song {} saved'.format(i, num_segments, filename))

            except Exception as e:
                print(e)

    return (list_segments, sr), filename


def split_audio(signal, top_db, frame_length=2048, hop_length=512):
    """
    Removes the segments of a signal with amplitude under
    a specified threshold.

        Parameters
        ----------
        signal : numpy array
            The input signal to obtain the chroma from
        top_db : int
            The amplitude threshold in decibels
        frame_length : int
            The size of the frames into which the signal is split
        hop_length : int
            The number of samples between each successive frame

    """
    import numpy as np
    import librosa

    if len(signal) > 0:
        edges = librosa.effects.split(
            signal,
            top_db=top_db,
            ref=np.max(signal),
            frame_length=frame_length,
            hop_length=hop_length
        )
        x_split = np.concatenate([signal[edge[0]:edge[1]] for edge in edges])
    else:
        x_split = signal

    return x_split


def create_chroma(signal, sr, n_fft):
    """
    Computes the chroma of a signal.

        Parameters
        ----------
        signal : numpy array
            The input signal to obtain the chroma from
        sr : int
            The sampling rate of the STFT needed for the chroma
        n_fft : int
            The size of the FFT needed for the STFT

    """
    import numpy as np
    import librosa

    S = np.abs(librosa.stft(signal, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    return chroma


def autocorrelate(signal, n_ite, resample=False, N=None):
    """
    Computes the autocorrelation of a signal.

        Parameters
        ----------
        signal : numpy array
            The input signal to autocorrelate
        n_ite : int
            The number of passes of autocorrelation
        resample : bool
            The option of resampling the input signal
        N : int
            The size of the resampled signal

    """
    import numpy as np

    if resample: signal = np.resample(signal, N)
    frequencies = np.fft.rfft(signal, axis=0)
    for _ in range(n_ite):
        ac = frequencies * np.conj(frequencies)
        frequencies = ac
    ac = np.fft.irfft(frequencies, axis=0)
    # normalize resulting signal
    ac_normalized = ac / np.max(ac) if np.max(ac) > 0 else ac

    return ac_normalized




"""
The rest of the methods below have been refactored from https://github.com/dr-costas/mad-twinnet
"""

def stft(x, windowing_func, fft_size, hop_size):
    import numpy as np

    window_size = windowing_func.size

    x = np.append(np.zeros(3 * hop_size), x)
    x = np.append(x, np.zeros(3 * hop_size))

    p_in = 0
    p_end = x.size - window_size
    indx = 0

    if np.sum(windowing_func) != 0.:
        windowing_func = windowing_func / np.sqrt(fft_size)

    xm_x = np.zeros((int(len(x) / hop_size), int(fft_size / 2) + 1), dtype=np.float32)
    xp_x = np.zeros((int(len(x) / hop_size), int(fft_size / 2) + 1), dtype=np.float32)

    while p_in <= p_end:
        x_seg = x[p_in:p_in + window_size]

        mc_x, pc_x = _dft(x_seg, windowing_func, fft_size)

        xm_x[indx, :] = mc_x
        xp_x[indx, :] = pc_x

        p_in += hop_size
        indx += 1

    return xm_x, xp_x

def _dft(x, windowing_func, fft_size):
    import numpy as np
    from scipy import fftpack

    half_n = int(fft_size / 2) + 1

    hw_1 = int(np.floor((windowing_func.size + 1) / 2))
    hw_2 = int(np.floor(windowing_func.size / 2))

    win_x = x * windowing_func

    fft_buffer = np.zeros(fft_size)
    fft_buffer[:hw_1] = win_x[hw_2:]
    fft_buffer[-hw_2:] = win_x[:hw_2]

    x = fftpack.fft(fft_buffer)

    magn_x = (np.abs(x[:half_n]))
    phase_x = (np.angle(x[:half_n]))

    return magn_x, phase_x


def i_stft(magnitude_spect, phase, window_size, hop):
    import numpy as np

    rs = _gl_alg(window_size, hop, (window_size - 1) * 2)

    hw_1 = int(np.floor((window_size + 1) / 2))
    hw_2 = int(np.floor(window_size / 2))

    # Acquire the number of STFT frames
    nb_frames = magnitude_spect.shape[0]

    # Initialise output array with zeros
    time_domain_signal = np.zeros(nb_frames * hop + hw_1 + hw_2)

    # Initialise loop pointer
    pin = 0

    # Main Synthesis Loop
    for index in range(nb_frames):
        # Inverse Discrete Fourier Transform
        y_buf = _i_dft(magnitude_spect[index, :], phase[index, :], window_size)

        # Overlap and Add
        time_domain_signal[pin:pin + window_size] += y_buf * rs

        # Advance pointer
        pin += hop

    # Delete the extra zeros that the analysis had placed
    time_domain_signal = np.delete(time_domain_signal, range(3 * hop))
    time_domain_signal = np.delete(
        time_domain_signal,
        range(time_domain_signal.size - (3 * hop + 1),
              time_domain_signal.size)
    )

    return time_domain_signal


def _gl_alg(window_size, hop_size, fft_size):
    import numpy as np
    from scipy import signal

    syn_w = signal.hamming(window_size) / np.sqrt(fft_size)
    syn_w_prod = syn_w ** 2.
    syn_w_prod.shape = (window_size, 1)
    redundancy = int(window_size / hop_size)
    env = np.zeros((window_size, 1))

    for k in range(-redundancy, redundancy + 1):
        env_ind = (hop_size * k)
        win_ind = np.arange(1, window_size + 1)
        env_ind += win_ind

        valid = np.where((env_ind > 0) & (env_ind <= window_size))
        env_ind = env_ind[valid] - 1
        win_ind = win_ind[valid] - 1
        env[env_ind] += syn_w_prod[win_ind]

    syn_w = syn_w / env[:, 0]

    return syn_w


def _i_dft(magnitude_spect, phase, window_size):
    import numpy as np
    from scipy import fftpack

    # Get FFT Size
    fft_size = magnitude_spect.size
    fft_points = (fft_size - 1) * 2

    # Half of window size parameters
    hw_1 = int(np.floor((window_size + 1) / 2))
    hw_2 = int(np.floor(window_size / 2))

    # Initialise output spectrum with zeros
    tmp_spect = np.zeros(fft_points, dtype=complex)
    # Initialise output array with zeros
    time_domain_signal = np.zeros(window_size)

    # Compute complex spectrum(both sides) in two steps
    tmp_spect[0:fft_size] = magnitude_spect * np.exp(1j * phase)
    tmp_spect[fft_size:] = magnitude_spect[-2:0:-1] * np.exp(-1j * phase[-2:0:-1])

    # Perform the iDFT
    fft_buf = np.real(fftpack.ifft(tmp_spect))

    # Roll-back the zero-phase windowing technique
    time_domain_signal[:hw_2] = fft_buf[-hw_2:]
    time_domain_signal[hw_2:] = fft_buf[:hw_1]

    return time_domain_signal


def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.

    :param sig: The input array.
    :type sig: numpy.core.multiarray.ndarray
    :param dtype: The desired data type of the output array, optional.
    :type dtype: (numpy.dtype)
    :return: The normalized floating point array.
    :rtype: (numpy.core.multiarray.ndarray, <dtype>)
    """
    import numpy as np


    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max

    return (sig.astype(dtype) - offset) / abs_max


def ffmpeg_decode(file_name, sr, mono=True):
    """Reads an audio file and returns its data. If `mono` is
    set to true, the returned audio data are monophonic.

    :param file_name: The file name of the audio file.
    :type file_name: str
    :param sr: The sample rate of the audio file, optional.
    :type sr: int
    :param mono: Get mono version, optional.
    :type mono: bool
    :return: The audio array.
    :rtype: (numpy.core.multiarray.ndarray, int)
    """
    import subprocess as sp
    import numpy as np

    command = [ 'ffmpeg',
            '-i', file_name,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ar', str(sr),
            '-ac', str(int(mono)), # stereo (set to '1' for mono)
            '-']
    pipe = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)
    stdoutdata = pipe.stdout.read()
    audio_array = np.fromstring(stdoutdata, dtype=np.int16)

    return pcm2float(audio_array)