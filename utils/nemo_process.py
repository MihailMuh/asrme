import sys

from nemo.collections.asr.models import NeuralDiarizer

from helpers import create_config


def diarize(temp_dir, device):
    msdd_model = NeuralDiarizer(cfg=create_config(temp_dir)).to(device)
    print(msdd_model.diarize())


if __name__ == '__main__':
    diarize(*sys.argv[1:])
