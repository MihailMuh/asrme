import sys

from nemo.collections.asr.models import NeuralDiarizer

from helpers import create_config


def diarize(device):
    msdd_model = NeuralDiarizer(cfg=create_config("/tmp/asrme_concatenated")).to(device)
    msdd_model.diarize()


if __name__ == '__main__':
    diarize(sys.argv[1])
