""""
This script performs the first preprocessing step: Vocals Isolation.
It uses the MadTwinNet (https://github.com/dr-costas/mad-twinnet)
as a source-separation algorithm, yielding close to SOTA results
at the moment of writing this code.
"""


def isolate_vocals(
        d00sr01_f1,
        weights_d0,
        rnn_enc_output_dim,
        original_input_dim,
        reduced_dim,
        context_length,
        window_size,
        fft_size,
        hop_size,
        seq_length,
        batch_size,
        top_db,
):
    """
    Isolates the vocals from an arbitrary music recording.

        Parameters
        ----------
        d00sr01_f1 : tuple
            The tuple containing the input music recording
        weights_d0 : int
            The already loaded in RAM weights of the neural network
        rnn_enc_output_dim : int
            The output dimension of the RNN encoder
        original_input_dim : int
            The original input dimension of the RNN encoder
        reduced_dim : int
            The reduced input dimension of the RNN encoder
        context_length : int
            The length of the scope considered by the RNN encoder
        window_size : int
            The size of the Hamming window applied to mitigate
            the leakage effect when performing the FFT
        fft_size : int
            The size of the FFT window used to compute the
            STFT and extract the magnitudes of the input audios
        hop_size : int
            The number of samples between each successive FFT window
        seq_length : int
            The lenght of the input audio fed to the neural network
        batch_size : int
            The size of the batches fed to the neural network
        top_db : int
            The threshold (in dB) above which
    """

    import os
    import time

    import numpy as np

    import torch
    torch.set_grad_enabled(False)

    from utils.data_utils import data_generator_testing, data_process_results_testing
    from utils.settings import debug, usage_output_string_per_example, \
        usage_output_string_total
    from madtwinnet.modules import RNNEnc, RNNDec, FNNMasker, FNNDenoiser


    # extract sampling rate and filename from the input tuple
    sr = d00sr01_f1[0][1]
    filename = d00sr01_f1[1]

    if debug:
        print('\n-- Cannot proceed in debug mode.',
              'Please set debug=False at the settings file.')
        print('-- Exiting.')
        exit(-1)
    print('\nNow MaD TwinNet will extract the voice from the provided files')

    # set the code to run on a GPU if present
    device = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'

    # masker modules
    rnn_enc = RNNEnc(reduced_dim, context_length, debug)
    rnn_dec = RNNDec(rnn_enc_output_dim, debug)
    fnn = FNNMasker(rnn_enc_output_dim, original_input_dim, context_length)

    # denoiser modules
    denoiser = FNNDenoiser(original_input_dim)

    rnn_enc.load_state_dict(weights_d0[0])
    rnn_enc.to(device).eval()

    rnn_dec.load_state_dict(weights_d0[1])
    rnn_dec.to(device).eval()

    fnn.load_state_dict(weights_d0[2])
    fnn.to(device).eval()

    denoiser.load_state_dict(weights_d0[3])
    denoiser.to(device).eval()

    mix, mix_magnitude, mix_phase = data_generator_testing(
        window_size=window_size,
        fft_size=fft_size,
        hop_size=hop_size,
        seq_length=seq_length,
        context_length=context_length,
        batch_size=batch_size,
        d00sr01_f1=d00sr01_f1
    )

    print('Let\'s go!\n')
    total_time = 0

    s_time = time.time()
    # initialize vocals array
    vocals_mag = np.zeros(
        (
            mix_magnitude.shape[0],
            seq_length - context_length * 2,
            window_size
        ),
        dtype=np.float32
    )

    # inference loop
    for batch in range(int(mix_magnitude.shape[0] / batch_size)):
        b_start = batch * batch_size
        b_end = (batch + 1) * batch_size

        # forward pass
        v_in = torch.from_numpy(mix_magnitude[b_start:b_end, :, :]).to(device)
        tmp_voice_predicted = rnn_enc(v_in)
        tmp_voice_predicted = rnn_dec(tmp_voice_predicted)
        tmp_voice_predicted = fnn(tmp_voice_predicted, v_in)
        tmp_voice_predicted = denoiser(tmp_voice_predicted)

        # get the magnitude of the resulting vocals
        vocals_mag[b_start:b_end, :, :] = tmp_voice_predicted.data.cpu().numpy()

    # recover the audio from the magnitudes
    _, vocals_wav_split = data_process_results_testing(
        voice_predicted=vocals_mag,
        window_size=window_size,
        mix_magnitude=mix_magnitude,
        mix_phase=mix_phase,
        hop=hop_size,
        context_length=context_length,
        top_db=top_db
    )

    e_time = time.time()

    # report time taken
    print(
        usage_output_string_per_example.format(
        f=os.path.basename(filename),
        t=e_time - s_time
        )
    )
    total_time += e_time - s_time
    print(usage_output_string_total.format(t=total_time))

    return (None, None, vocals_wav_split, sr), filename


# since PySpark needs an iter() object, we create a wrapper method for that purpose
def P_isolate_vocals(
        P_d00sr01_f1, weights_d0,
        rnn_enc_output_dim,
        original_input_dim,
        reduced_dim,
        context_length,
        window_size,
        fft_size,
        hop_size,
        seq_length,
        batch_size,
        top_db
):

    P_out = []

    for d00sr01_f1 in P_d00sr01_f1:
        P_out.append(
            isolate_vocals(
                d00sr01_f1,
                weights_d0,
                rnn_enc_output_dim,
                original_input_dim,
                reduced_dim,
                context_length,
                window_size,
                fft_size,
                hop_size,
                seq_length,
                batch_size,
                top_db
            )
        )

    return iter(P_out)
