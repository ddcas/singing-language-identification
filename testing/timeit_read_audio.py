"""
This script compares the performance of different audio
reading methods from diverse Python libraries
"""

import timeit
import argparse


def test_performance(args):
    # read audio using ffmpeg as a subprocess
    setup_ffmpeg = '''
    
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
        """Reads an audio file and returns its data. If `mono` is \
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
        pipe = sp.Popen(
            command,
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            bufsize=10**8
            )
        stdoutdata = pipe.stdout.read()
        audio_array = np.fromstring(stdoutdata, dtype=np.int16)
    
        return pcm2float(audio_array)
    
    '''

    # read audio using Soundfile.read
    setup_sf = '''
    
    def sf_read(f, sr):
        import soundfile as sf
    
        try:
            data = sf.read(f, dtype='float32')
        except Exception as e:
            print('Exception caught:', e)
            print('File responsible:', f)
            data = None, None
    
        return data
    '''

    # read audio using librosa.load
    setup_librosa = '''
    
    def librosa_load(f, sr):
        import librosa
    
        try:
            data = librosa.load(f, sr, mono=True)  # sr=None to keep native sr
        except Exception as e:
            print('Exception caught:', e)
            print('File responsible:', f)
            data = None, None
    
        return data
    '''
    n = args.num_iterations
    f = args.filename
    sr = args.sampling_rate
    print(
        'FFMPEG:',
        min(
            timeit.Timer(
                'ffmpeg_decode(\'{}\', {})'.format(f, sr),
                setup=setup_ffmpeg).repeat(1, n)
        )
    )
    print(
        'SF:',
        min(
            timeit.Timer(
                'sf_read(\'{}\', {})'.format(f, sr),
                setup=setup_sf).repeat(1, n)
        )
    )
    print(
        'LIBROSA:',
        min(
            timeit.Timer(
                'librosa_load(\'{}\', {})'.format(f, sr),
                setup=setup_librosa).repeat(1, n)
        )
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of times to run the test methods')
    parser.add_argument('--filename', type=str, default='',
                        help='Filename to test')
    parser.add_argument('--sampling_rate', type=int, default=16000,
                        help='Sampling rate of the audios that are read')

    parse_args, unknown = parser.parse_known_args()

    test_performance(parse_args)