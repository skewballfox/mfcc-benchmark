from librosa.feature import melspectrogram  as librosa_mel
from speechsauce import mel_spectrogram as speechsauce_mel

frame_length = 0.02
frame_stride = 0.01
num_filters = 40
sample_rate = 16000
low_freq = 0
hi_freq = 8000
num_cepstral = 13

mel_spec_functions = {
    "speechsauce": lambda signal,sample_rate,hop_t,fft_size: speechsauce_mel(signal,
                                                sample_rate,
        frame_length=frame_length,
        num_cepstral=num_cepstral,
        frame_stride=hop_t,
        num_filters=num_filters,
        fft_length=512,
        low_frequency=0,
        dc_elimination=True,
        high_frequency=None,
        ),
    "librosa": lambda signal,hop_t,fft_size: librosa_mel(signal,)
}