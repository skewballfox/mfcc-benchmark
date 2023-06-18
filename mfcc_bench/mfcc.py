import time

from speechpy.feature import mfcc as speechpy_mfcc
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
num_cepstral = 13

mfcc_functions = {
    "speechpy": lambda signal: speechpy_mfcc(
        signal,
        sampling_frequency=sample_rate,
        frame_length=frame_length,
        num_cepstral=num_cepstral,
        frame_stride=frame_stride,
        num_filters=num_filters,
        fft_length=512,
        low_frequency=0,
        high_frequency=None,
    ),
    "speechsauce": lambda signal: speechpy_mfcc(
        signal,
        sampling_frequency=sample_rate,
        frame_length=frame_length,
        num_cepstral=num_cepstral,
        frame_stride=frame_stride,
        num_filters=num_filters,
        fft_length=512,
        low_frequency=0,
        high_frequency=None,
    ),
    "sonopy": lambda signal: sonopy_mfcc(
        audio=signal,
        sample_rate=sample_rate,
        window_stride=(
            int(frame_length * sample_rate),
            int(frame_stride * sample_rate),
        ),
        fft_size=512,
        num_filt=num_filters,
        num_coeffs=num_cepstral,
    ),
    "librosa": lambda signal: librosa_mfcc(
        y=signal,
        sr=sample_rate,
        hop_length=int(frame_length * sample_rate),
        n_mels=num_filters,
        n_mfcc=num_cepstral,
        n_fft=512,
        fmin=low_freq,
        fmax=hi_freq,
        norm=None,  # type: ignore
    ),
}


def bench_mfccs(signal: NDArray[np.float32], Warmup=False):
    results = {}
    for k in mfcc_functions:
        if Warmup:
            for _ in range(10):
                _ = mfcc_functions[k](signal)
        start = time.time()
        _ = mfcc_functions[k](signal)
        results[k] = time.time() - start
    return results
