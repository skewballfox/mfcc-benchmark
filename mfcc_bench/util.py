import sys
import soundfile as sf
from pathlib import Path
import numpy as np
import requests


def get_test_file_path(data_dir: Path, url:str) -> Path | None:
    """function to get either return the file path to a input audio file, downloads it if not locally available"""
    data_dir.mkdir(exist_ok=True)
    file_name=data_dir/url[url.rfind("/")+1::]
    print(file_name)
    if file_name.is_file():
        return file_name
    else:
        print('starting download')
        headers = {
            'User-Agent': 'My User Agent 1.0'
        }
        r= requests.get(url,allow_redirects=True,headers=headers)
        if r.status_code==200:
            with open(file_name,"wb") as w:
                w.write(r.content)
            return file_name
        else:
            print(f"r status{r.status_code}")
            sys.exit()

def audio_to_array(audio_file_path: Path):
    data, samplerate = sf.read(str(audio_file_path),always_2d=False)
    return data
