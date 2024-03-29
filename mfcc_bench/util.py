import sys
from typing import Optional, Tuple
import soundfile as sf
from pathlib import Path
import numpy as np
import requests
from numpy.typing import NDArray
from numpy import floating, float32

def get_test_file_path(data_dir: Path, url: str) -> Path:
    """function to get either return the file path to a input audio file, downloads it if not locally available"""
    data_dir.mkdir(exist_ok=True)
    file_name = data_dir / url[url.rfind("/") + 1 : :]
    print(file_name)
    if file_name.is_file():
        return file_name
    else:
        print("starting download")
        headers = {"User-Agent": "My User Agent 1.0"}
        r = requests.get(url, allow_redirects=True, headers=headers)
        if r.status_code != 200:
            print(f"problem with file {file_name} with url: {url}")
            print(f"r status{r.status_code}")
            sys.exit()
        with open(file_name, "wb") as w:
            w.write(r.content)
        return file_name


def audio_to_array(audio_file_path: Path) -> Tuple[NDArray[float32],int]:
    data, samplerate = sf.read(str(audio_file_path),dtype='float32', always_2d=False)
    print(f"samplerate is {samplerate}")
    return data, samplerate

def dual_channel_to_mono(data: NDArray[float32])->NDArray[float32]:
    return data.sum(axis=1) / 2
