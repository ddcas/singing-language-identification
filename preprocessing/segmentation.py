""""
This script performs the second preprocessing step: Segmentation.
It uses a combination of signal processing algorithms (chroma,
autocorrelation, energy level) to extract those segments of
a music recording that are more likely to contain the vocals.
"""


def find_relevant_segments(
        vws00sr01_f1,
        n_seg,
        seg_len_sec,
        size_fft_chroma,
        n_ite_autocorrelation
):
    """
    Finds 'n_segments' of length 'seg_len_sec' that are the
    the most likely to contain vocals.

        Parameters
        ----------
        vws00sr01_f1 : tuple
            The tuple containing the isolated vocals
        n_seg : int
            The number of segments to extract from the vocals
        seg_len_sec : int
            The length of each extracted segment
        size_fft_chroma : int
            The size of the FFT window computed for the chroma
        n_ite_autocorrelation : int
            The number of autocorrelation passes
    """
    import numpy as np
    import librosa

    from utils.audio_utils import create_chroma, autocorrelate
    from utils.settings import hyper_parameters

    # separate variables from the initial tuple
    song_vocals_wave = vws00sr01_f1[0][0]
    sr = vws00sr01_f1[0][1]
    filename = vws00sr01_f1[1]

    print('\nFinding relevant segments...')
    print(song_vocals_wave.shape, sr, size_fft_chroma)
    if song_vocals_wave.shape[0] == 0:
        return ([], sr), filename

    # compute chroma
    chroma = create_chroma(song_vocals_wave, sr, size_fft_chroma)
    print('\nChroma completed')

    # compute autocorrelation
    ac = autocorrelate(chroma, n_ite_autocorrelation)
    print('\nAutocorrelation completed')
    num_samples_chroma = chroma.shape[1]
    song_len_sec = len(song_vocals_wave) / sr
    chroma_sr = num_samples_chroma / song_len_sec
    chroma_clip_len_samples = seg_len_sec * chroma_sr

    # extract initial chorus candidates
    chroma_choruses = np.mean(ac, axis=0).argsort()[-n_seg:][::-1]
    best_choruses = chroma_choruses[~(np.triu(np.abs(chroma_choruses[:, None] -
                    chroma_choruses) <= chroma_clip_len_samples, 1)).any(0)]
    for chorus in best_choruses:
        chorus_sec = chorus * song_len_sec / num_samples_chroma
        print(
            '-- In file {0}: relevant segment',
            'found at {1:g} min {2:.2f} sec'.format(
                filename,
                chorus_sec // 60,
                chorus_sec % 60,
                chorus
            )
        )

    print(
        'Finished processing file {}.',
        'Number of relevant segments found: {}'.format(
            filename,
            len(best_choruses)
        )
    )
    if best_choruses is None:
        raise ValueError(
            '\n\nNo choruses where found :(' + \
            'Try increasing the input audio length.'
        )

    list_songs_choruses = []
    for i, chorus_start in enumerate(best_choruses):
        chorus_start_sample = chorus_start / chroma_sr
        chorus_wave_data = song_vocals_wave[int(chorus_start_sample * sr) : \
                            int((chorus_start_sample + seg_len_sec) * sr)]
        if len(chorus_wave_data) == seg_len_sec*sr:
            list_songs_choruses.append(chorus_wave_data)

    # "flatnergy" gives a score for the "voice-likeness" of a segment
    list_flatnergy_avg = []
    for seg in list_songs_choruses:
        frames_energies = np.array(
            [
                sum(abs(seg[i:i + hyper_parameters['fft_size']]) ** 2)
                    for i in range(0, len(seg), hyper_parameters['hop_size'])
            ]
        )
        frames_flatnesses = frames_energies* \
                            librosa.feature.spectral_flatness(
                                seg, n_fft=hyper_parameters['fft_size'],
                                hop_length=hyper_parameters['hop_size']
                            )
        list_flatnergy_avg.append(
            np.mean(frames_energies) * (1 - np.mean(frames_flatnesses))
        )

    # sort clips by energy*(1-spectral_flatness)
    list_song_clips =[
        seg for _, seg in sorted(
            zip(
                list_flatnergy_avg,
                list_songs_choruses
            )
        )
     ][-hyper_parameters['cliper_song']:]



    print(
        'Removing noisy clips and clips shorter',
        'than the specified length ({} seconds)...'.format(seg_len_sec)
    )
    print(
        'All files processed. Final number of',
        'segments found: {}\n'.format(len(list_song_clips))
    )

    return (list_song_clips, sr), filename


# since PySpark needs an iter() object, we create a wrapper method
def P_find_relevant_segments(
        P_vm00vw01vws02sr03_f1,
        n_segments,
        segment_len_sec,
        size_fft_chroma,
        n_ite_autocorrelation
):

    P_out = []
    for vm00vw01vws02sr03_f1 in P_vm00vw01vws02sr03_f1:
        P_out.append(
            find_relevant_segments(
                (
                    (
                        vm00vw01vws02sr03_f1[0][2],
                        vm00vw01vws02sr03_f1[0][3]
                    ),
                    vm00vw01vws02sr03_f1[1]
                ),
                n_segments, segment_len_sec,
                size_fft_chroma,
                n_ite_autocorrelation
            )
        )

    return iter(P_out)