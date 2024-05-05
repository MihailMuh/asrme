import numpy as np


class TranscribationRequest:
    def __init__(self, temp_dir, audio_file_name: str, audio_file: np.ndarray):
        self.temp_dir = temp_dir
        self.audio_file_name = audio_file_name
        self.audio_file = audio_file
        self.audio_length = len(audio_file) / 16_000
