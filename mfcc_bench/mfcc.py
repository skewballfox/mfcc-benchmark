import time

from speechpy.feature import mfcc as speechpy_mfcc
from speechsauce import mfcc as speechsauce_mfcc
from sonopy import mfcc_spec as sonopy_mfcc
from librosa.feature import mfcc as librosa_mfcc
from numpy.typing import NDArray
import numpy as np

frame_length = 0.02
frame_stride = 0.01
num_filters = 40
sample_rate = 16000
low_freq = 0
hi_freq = 8000
num_coeffs = 13
"""num_cepstrals, n_mfcc, num_coeffs are all the same thing"""

mfcc_functions = {
    "speechpy": lambda signal,sample_rate,hop_t,fft_size: speechpy_mfcc(
        signal,
        sampling_frequency=sample_rate,
        frame_length=frame_length,
        num_cepstral=num_coeffs,
        frame_stride=hop_t,
        num_filters=num_filters,
        fft_length=fft_size,
        low_frequency=0,
        high_frequency=hi_freq,
    ),
    "speechsauce": lambda signal,sample_rate,hop_t,fft_size: speechsauce_mfcc(
        signal,
        sampling_frequency=sample_rate,
        frame_length=frame_length,
        num_cepstral=num_coeffs,
        frame_stride=hop_t,
        num_filters=num_filters,
        fft_length=fft_size,
        low_frequency=0,
        dc_elimination=True,
        high_frequency=hi_freq,
    ),
    "sonopy": lambda signal,sample_rate,hop_t,fft_size: sonopy_mfcc(
        audio=signal,
        sample_rate=sample_rate,
        window_stride=(
            int(hop_t * sample_rate),
            int(hop_t * sample_rate),
        ),
        fft_size=fft_size,
        num_filt=num_filters,
        num_coeffs=num_coeffs,
    ),
    "librosa": lambda signal,sample_rate,hop_t,fft_size: librosa_mfcc(
        y=signal,
        sr=sample_rate,
        hop_length=int(hop_t * sample_rate),
        n_mels=num_filters,
        n_mfcc=num_coeffs,
        n_fft=fft_size,
        fmin=low_freq,
        fmax=hi_freq,
        norm=None,  # type: ignore
    ),
}


def bench_mfccs(signal: NDArray[np.float32],sample_rate,hop_t,fft_size, Warmup=False):
    results = {}
    for k in mfcc_functions:
        if Warmup:
            for _ in range(10):
                _ = mfcc_functions[k](signal,sample_rate,hop_t,fft_size)
        start = time.time()
        _ = mfcc_functions[k](signal,sample_rate,hop_t,fft_size)
        results[k] = time.time() - start
    return results
