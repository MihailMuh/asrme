import logging

from nemo.collections.asr.models import NeuralDiarizer

from utils.helpers import *


class DiarizerService:
    def __init__(self):
        self.__init_logger()

        self.__device: str = "cuda"

        self.msdd_model = NeuralDiarizer(cfg=create_config("/tmp/asrme_concatenated")).to(self.__device)
        self.__logger.debug("DiarizerService initialized!")

    async def diarize(self, audio_lengths: list[int]) -> list[list[list[int]]]:
        self.msdd_model.diarize()
        return split_nemo_result(await read_nemo_result("/tmp/asrme_concatenated"), audio_lengths)

    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
