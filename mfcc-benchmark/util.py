import soundfile as sf
from pathlib import Path
import numpy as np

def ogg_to_array(audio_file_path: Path):
    data, samplerate = sf.read(str(audio_file_path),always_2d=False)
    print(data.shape)
    return data
