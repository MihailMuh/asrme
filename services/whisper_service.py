import logging
import time
import uuid
from itertools import chain

import torch

import whisper_s2t
from dto.transcribation_request import TranscribationRequest
from utils.helpers import *
from whisper_s2t.backends.ctranslate2.model import WhisperModelCT2


class WhisperService:
    def __init__(self):
        self.__init_logger()

        self.__arch: str = "large-v2"  # large-v3 produces MORE hallucinations
        self.__device: str = "cuda"
        self.__batch_size: int = 128
        self.__compute_type: str = "bfloat16"
        self.__language: str = "ru"
        self.__asr_options: dict = {
            "beam_size": 5,
            "patience": 2,
            "word_timestamps": True,
            "return_scores": False,
            "return_no_speech_prob": False,
            "sampling_temperature": 1,
        }

        self.__tasks: list[tuple[uuid, TranscribationRequest]] = []
        self.__results: dict[uuid, str] = {}
        self.__in_process: bool = False
        self.__nemo_process_py: Path = Path(__file__).parent.parent / "utils" / "nemo_process.py"

        os.system(f"rm -rf /tmp/asrme_*")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.__logger.debug("Whisper initialization...")
        self.__whisper: WhisperModelCT2 = whisper_s2t.load_model(
            self.__arch,
            compute_type=self.__compute_type,
            asr_options=self.__asr_options
        )

        self.__logger.debug(f"WhisperService initialized!")

    async def create_task(self, audio_bytes: bytes) -> uuid.UUID:
        async def add_task() -> uuid.UUID:
            id_response: uuid.UUID = uuid.uuid1()
            self.__tasks.append(
                (id_response, TranscribationRequest(*(await load_audio(str(id_response), audio_bytes)))))
            return id_response

        while self.__in_process:
            await asyncio.sleep(0.100)

        if len(self.__tasks) == 4:
            self.__in_process = True

            self.__logger.info(f"{len(self.__tasks) + 1} tasks caught!")
            time_start = time.time()

            id_r: uuid.UUID = await add_task()
            await self.__transcribe()

            self.__logger.info(f"Pipeline took {time.time() - time_start} seconds")

            self.__tasks = []
            self.__in_process = False
            return id_r

        return await add_task()

    def get_task(self, id_request: uuid.UUID) -> str:
        return self.__results.pop(id_request, "")

    async def __transcribe(self):
        self.__logger.debug("Concatenating audios...")
        await concat_audio([id_t_request[1].temp_dir for id_t_request in self.__tasks])

        self.__logger.debug("Creating the nemo diarization process...")
        nemo_process = await asyncio.create_subprocess_exec(*[
            "python", self.__nemo_process_py, self.__device
        ], stdout=DEVNULL, stderr=DEVNULL)

        self.__logger.debug("Creating transcribation...")
        transcribed_results: list[list[dict]] = self.__whisper.transcribe_with_vad(
            [id_t_request[1].audio_file for id_t_request in self.__tasks],
            batch_size=self.__batch_size, lang_codes=[self.__language], tasks=["transcribe"]
        )

        self.__logger.debug("Waiting for the nemo diarization process finish...")
        await nemo_process.wait()
        speaker_tss: list[list[list[int]]] = split_nemo_result(
            await read_nemo_result("/tmp/asrme_concatenated"),
            [int(id_t_request[1].audio_length * 1000) for id_t_request in self.__tasks]
        )

        self.__logger.debug("Calculating the results...")
        for i, result in enumerate(transcribed_results):
            os.system(f"rm -rf {self.__tasks[i][1].temp_dir}/")
            self.__results[self.__tasks[i][0]] = detect_admin_and_patient(
                assign_diarization_to_transcribation(
                    speaker_tss[i],
                    list(chain.from_iterable(seg["word_timestamps"] for seg in result))
                )
            )

    def dispose(self):
        del self.__whisper
        torch.cuda.empty_cache()

        os.system(f"rm -rf /tmp/asrme_*")

    def __init_logger(self):
        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(logging.DEBUG)
