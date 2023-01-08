from pathlib import Path
from mfcc_bench.util import audio_to_array, get_test_file_path
from mfcc_bench.mfcc import bench_mfccs, mfcc_functions
import numpy as np
import math
import time

data_path = Path(__file__).parent.parent / "data"

quick_fox_mp3 = get_test_file_path(
    data_path,
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a1/Audio_Sample_-_The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog.ogg/Audio_Sample_-_The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog.ogg.mp3",
)
dead_presidents = get_test_file_path(
    data_path,
    "https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",
)
print(quick_fox_mp3)
data = audio_to_array(quick_fox_mp3).T[0]

print(f"data len is {len(data)}")

frame_length = 0.02
frame_stride = 0.01
num_filters = 40
sample_rate = 16000
low_freq = 0
hi_freq = 8000
num_cepstral = 13


bench_results = bench_mfccs(data, Warmup=True)
for k in bench_results:
    print(f"{k} took: {bench_results[k]}")
pie = mfcc_functions["speechpy"](data)
sauce = mfcc_functions["speechsauce"](data)
# get the relative difference between the two
print(np.linalg.norm(pie - sauce))
print(np.allclose(pie, sauce))

print(pie[100:110])
print(sauce[100:110])
