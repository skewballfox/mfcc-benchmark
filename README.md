# MFCC Benchmark

a simple python module for comparing the various libraries for speech preprocessing. Primarily used in the development of [speechsauce](https://github.com/secretsauceai/mfcc-rust) (name pending, please suggest literally anything else)

## Getting Started

setting up the virtual environment is pretty seamless if you have poetry installed:

```shell
git clone <insert URL>
cd mfcc-benchmark
poetry install
poetry shell
```

afterwards make sure to download the data for testing.

```shell
mkdir data
cd data
wget https://upload.wikimedia.org/wikipedia/commons/5/50/Jfk_rice_university_we_choose_to_go_
to_the_moon.ogg
cd ..
```

Note I'm hoping to add a cli to this project, and more files for testing. but until then you'll have to download the files manually.
