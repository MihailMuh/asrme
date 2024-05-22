import logging
import time
import uuid
from itertools import chain

import torch

import whisper_s2t
from dto.transcribation_request import TranscribationRequest
from services.diarizer_service import DiarizerService
from utils.helpers import *
from whisper_s2t.backends.ctranslate2.model import WhisperModelCT2


class WhisperService:
    def __init__(self, send_transcribe_func):
        self.__init_logger()

        self.__send_transcribe_func = send_transcribe_func

        self.__arch: str = "large-v2"  # large-v3 produces MORE hallucinations
        self.__device: str = "cuda"
        self.__batch_size: int = 32
        self.__compute_type: str = "int8"
        self.__language: str = "ru"
        self.__asr_options: dict = {
            "beam_size": 1,
            "patience": 2,
            "word_timestamps": True,
            "return_scores": False,
            "return_no_speech_prob": False,
            "sampling_temperature": 1,
            "word_aligner_model": "tiny",
        }

        self.__tasks: list[tuple[uuid, TranscribationRequest]] = []
        self.__total_audios_length: int = 0
        self.__in_process: bool = False

        os.system(f"rm -rf /tmp/asrme_*")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.__logger.debug("Whisper initialization...")
        self.__whisper: WhisperModelCT2 = whisper_s2t.load_model(
            self.__arch,
            compute_type=self.__compute_type,
            asr_options=self.__asr_options
        )

        self.__logger.debug("DiarizerService initialization...")
        self.__diarizer_service = DiarizerService()

        self.__logger.debug(f"WhisperService initialized!")

    async def create_task(self, audio_bytes: bytes, force: bool = False) -> uuid.UUID:
        async def add_task() -> uuid.UUID | None:
            id_response: uuid.UUID = uuid.uuid1()
            request = TranscribationRequest(*(await load_audio(str(id_response), audio_bytes)))

            if self.__total_audios_length + request.audio_length >= 6000:
                return None

            self.__tasks.append((id_response, request))
            self.__total_audios_length += request.audio_length

            return id_response

        while self.__in_process:
            await asyncio.sleep(0.300)

        if force:
            self.__in_process = True

            self.__logger.info(f"{len(self.__tasks) + 1} tasks with total length {self.__total_audios_length} caught!")
            time_start = time.time()

            try:
                await self.__transcribe()
                self.__logger.info(f"Pipeline took {time.time() - time_start} seconds")
            except Exception as exception:
                self.__logger.error(exception)

            self.__tasks = []
            self.__total_audios_length = 0
            self.__in_process = False

        id = await add_task()
        if not id:
            return await self.create_task(audio_bytes, True)
        return id

    async def __transcribe(self):
        self.__logger.debug("Concatenating audios...")
        await concat_audio([id_t_request[1].temp_dir for id_t_request in self.__tasks])

        self.__logger.debug("Creating the transcribation...")
        transcribed_results: list[list[dict]] = self.__whisper.transcribe_with_vad(
            [id_t_request[1].audio_file for id_t_request in self.__tasks],
            batch_size=self.__batch_size, lang_codes=[self.__language], tasks=["transcribe"]
        )

        self.__logger.debug("Creating the diarization...")
        speaker_tss: list[list[list[int]]] = await self.__diarizer_service.diarize(
            [int(id_t_request[1].audio_length * 1000) for id_t_request in self.__tasks]
        )

        self.__logger.debug("Calculating the results...")
        results = []
        for i, result in enumerate(transcribed_results):
            try:
                os.system(f"rm -rf {self.__tasks[i][1].temp_dir}/")

                assigned_speech: list[list[str]] = assign_diarization_to_transcribation(
                    speaker_tss[i],
                    list(chain.from_iterable(seg["word_timestamps"] for seg in result))
                )

                results.append(str(self.__tasks[i][0]) + "|||" + detect_admin_and_patient(assigned_speech))
            except Exception as exception:
                self.__logger.error(exception)

        await self.__send_transcribe_func(results)

    def dispose(self):
        del self.__whisper
        del self.__diarizer_service
        torch.cuda.empty_cache()

        os.system(f"rm -rf /tmp/asrme_*")

    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
