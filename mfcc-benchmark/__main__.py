from pathlib import Path
from .util import ogg_to_array
import requests

data_path=Path(__file__).parent.parent / "data"
dead_presidents=data_path / "Jfk_rice_university_we_choose_to_go_to_the_moon.ogg"
#if data_path.exists():
#    #data_path.mkdir()
#    headers = {
#    'User-Agent': 'My User Agent 1.0'
#    }
#    r=requests.get("https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_to_the_moon.ogg",headers=headers)
#    print(r)

dead_presidents=data_path / "Jfk_rice_university_we_choose_to_go_to_the_moon.ogg"
data= ogg_to_array(dead_presidents)
i =0
single_channel=data[0]

