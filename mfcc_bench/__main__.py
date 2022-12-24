from pathlib import Path
from mfcc_bench.util import audio_to_array,get_test_file_path
from speechpy.feature import mfcc as speechpy_mfcc
import numpy as np
import math
data_path=Path(__file__).parent.parent / "data"

quick_fox_mp3=get_test_file_path(data_path,"https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a1/Audio_Sample_-_The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog.ogg/Audio_Sample_-_The_Quick_Brown_Fox_Jumps_Over_The_Lazy_Dog.ogg.mp3")
dead_presidents=get_test_file_path(data_path,"https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg")

data= audio_to_array(quick_fox_mp3).T[0]
print(f"data len is {len(data)}")

frame_length = 0.02
frame_stride = 0.02
num_filters=40
fs = 16000

print(len(data)-int(math.ceil((len(data)-fs)/np.round(fs*frame_length))))

num_cepstral = 13

speechpy_call=lambda signal: speechpy_mfcc(signal, sampling_frequency=fs,
                            frame_length=0.020, num_cepstral=num_cepstral, frame_stride=0.01,
                            num_filters=num_filters, fft_length=512, low_frequency=0,
                            high_frequency=None)

speechsauce_call=lambda signal: speechpy_mfcc(signal, sampling_frequency=fs,
                            frame_length=0.020, num_cepstral=num_cepstral, frame_stride=0.01,
                            num_filters=num_filters, fft_length=512, low_frequency=0,
                            high_frequency=None)

speechpy_call(data)