import os

from platformdirs import user_cache_dir

from .utils import write_outputs

BASE_PATH = os.path.dirname(__file__)

CACHE_DIR = user_cache_dir("whisper_s2t")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_model(model_identifier="large-v2", **model_kwargs):
    # if model_identifier in ['large-v3']:
    #     model_kwargs['n_mels'] = 128

    from .backends.ctranslate2.model import WhisperModelCT2 as WhisperModel
    return WhisperModel(model_identifier, **model_kwargs)
