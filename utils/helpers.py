import asyncio
import json
import os
from asyncio.subprocess import Process, DEVNULL
from pathlib import Path

import aiofiles
import aiofiles.os as aios
import numpy as np
from omegaconf import OmegaConf


def create_config(output_dir):
    config = OmegaConf.load(Path(__file__).parent / "diar_infer_telephonic.yaml")

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": 10,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    return config


def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
    # if current word is the last word
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]

    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            # if next word doesn't have a start timestamp
            # merge it with the current word and delete it
            word_timestamps[current_word_index]["word"] += (
                    " " + word_timestamps[next_word_index]["word"]
            )

            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1
            if next_word_index == len(word_timestamps):
                return final_timestamp

        else:
            return word_timestamps[next_word_index]["start"]


def filter_missing_timestamps(word_timestamps, initial_timestamp=0, final_timestamp=None) -> list:
    if len(word_timestamps) == 0:
        return []

    # handle the first and last word
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = (
            initial_timestamp if initial_timestamp is not None else 0
        )
        word_timestamps[0]["end"] = _get_next_start_timestamp(
            word_timestamps, 0, final_timestamp
        )

    result = [
        word_timestamps[0],
    ]

    for i, ws in enumerate(word_timestamps[1:], start=1):
        # if ws doesn't have a start and end
        # use the previous end as start and next start as end
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)

        if ws["word"] is not None:
            result.append(ws)
    return result


def detect_admin_and_patient(text: list[list[str]]) -> str:
    projection: dict = dict(set((line[0], "") for line in text))

    for line in text:
        speaker: str = line[0]
        speech: str = line[1].strip()

        if (not speaker) or (not speech):
            continue

        if not projection[speaker]:
            speech_lower: str = speech.lower()
            if ("администратор" in speech_lower) or ("клиника" in speech_lower):
                projection[speaker] = "Администратор: "
            elif ("нам нужно" in speech_lower) or ("записаться" in speech_lower):
                projection[speaker] = "Пациент: "

    if "".join(projection.values()).count(":") == 1:  # if detect ONLY ONE person
        speaker_to_fill: str = "Пациент: " if "Администратор: " in projection.values() else "Администратор: "
        for speaker in projection.keys():
            if not projection[speaker]:
                projection[speaker] = speaker_to_fill

    result = ""
    for line in text:
        speaker: str = line[0]
        speech = line[1].strip()
        if not speech:
            continue

        result += projection.get(speaker, "") + speech + "\n"

    return result.strip()


async def load_audio(uid: str, audio_bytes: bytes) -> (str, str, np.ndarray):
    """
     Open an audio file and read as mono waveform, resampling as necessary

     Parameters
     ----------
     uid: str
         The name of temp dir to create

     audio_bytes: bytes
         The audio bytes to open process

     Returns
     -------
     A NumPy array containing the audio waveform, in float32 dtype.
     """

    temp_dir = f"/tmp/asrme_{uid}"
    await aios.makedirs(temp_dir, exist_ok=True)

    async with aiofiles.open(os.path.join(temp_dir, "original"), 'wb+') as audio_file:
        await audio_file.write(audio_bytes)
        await audio_file.flush()
        mono_file: str = os.path.join(temp_dir, "mono_file.wav")

        cmd: list[str] = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-threads", "0",
            "-i", audio_file.name,
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            mono_file
        ]

        process: Process = await asyncio.create_subprocess_exec(*cmd, stdout=DEVNULL, stderr=DEVNULL)
        await process.wait()

        async with aiofiles.open(mono_file, "rb") as file:
            result: np.ndarray = np.frombuffer(await file.read(), np.int16).flatten().astype(np.float32) / 32768.0

            return temp_dir, audio_file.name, result


async def concat_audio(audio_files_dirs: list[str]):
    async with aiofiles.tempfile.NamedTemporaryFile('w+', suffix=".txt") as txt:
        await txt.writelines(f"file {os.path.join(tempdir, 'mono_file.wav')}\n" for tempdir in audio_files_dirs)
        await txt.flush()

        await aios.makedirs("/tmp/asrme_concatenated", exist_ok=True)

        cmd: list[str] = [
            "ffmpeg",
            "-y",
            "-threads", "0",
            "-f", "concat",
            "-safe", "0",
            "-i", txt.name,
            "-c", "copy",
            "/tmp/asrme_concatenated/mono_file.wav"
        ]

        process: Process = await asyncio.create_subprocess_exec(*cmd, stdout=DEVNULL, stderr=DEVNULL)
        await process.wait()


def get_speakers_list(speaker_ts: list[list[int]]) -> list[int]:
    return list(map(lambda segment: segment[-1], speaker_ts))


async def read_nemo_result(temp_dir_name: str) -> list[list[int]]:
    speaker_ts: list[list[int]] = []
    async with aiofiles.open(os.path.join(temp_dir_name, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = await f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        return speaker_ts


def split_nemo_result(speaker_ts: list[list[int]], audio_files_lengths: list[int]) -> list[list[list[int]]]:
    result_split: list[list[list[int]]] = []
    current_audio_length: int = audio_files_lengths[0]
    j = 0
    current_segment_start_index: int = 0

    for i, segment in enumerate(speaker_ts):
        if segment[0] >= current_audio_length:
            j += 1
            current_audio_length += audio_files_lengths[j]
            result_split.append(speaker_ts[current_segment_start_index: i])
            current_segment_start_index = i

    result_split.append(speaker_ts[current_segment_start_index:])
    return result_split


def assign_diarization_to_transcribation(
        diarization: list[list[int]], transcribation_aligned: list[dict]) -> list[list[str]]:
    phrases: list[dict] = []

    current_phrase: str = ""
    current_phrase_start: int = 0
    current_phrase_end: int = 0

    for align in transcribation_aligned:
        word: str = align["word"]
        start: int = int(align["start"] * 1000)
        end: int = int(align["end"] * 1000)

        if not current_phrase:
            current_phrase_start = start
            current_phrase_end = end
            current_phrase += word + " "
            continue

        if (word.strip()[0].isupper()) and (current_phrase.strip()[-1] != ",") and (
                not current_phrase.strip()[-1].isalpha()):
            phrases.append({"start": current_phrase_start, "end": current_phrase_end, "text": current_phrase.strip()})
            current_phrase = word + " "
            current_phrase_start = start
            current_phrase_end = end
            continue

        current_phrase += word + " "
        current_phrase_end = end

    phrases.append({"start": current_phrase_start, "end": current_phrase_end, "text": current_phrase.strip()})

    offset: int = diarization[0][0] - phrases[0]["start"]
    diarization_normalized = []
    for segment in diarization:
        diarization_normalized.append([segment[0] - offset, segment[1] - offset, segment[2]])

    speaker_ts_normalized = [diarization_normalized[0]]
    for segment in diarization_normalized[1:]:
        if speaker_ts_normalized[-1][2] != segment[2]:
            speaker_ts_normalized.append(segment)
        else:
            speaker_ts_normalized[-1][1] = segment[1]

    is_one_speaker: bool = len(speaker_ts_normalized) == 1
    assigned: list[list[str]] = []

    for i in range(len(phrases)):
        phrase = phrases[i]
        start, end, text = phrase["start"], phrase["end"], phrase["text"]
        suitable_segments: list[list[int]] = []
        suitable_speakers: list[int] = []

        if is_one_speaker:
            assigned.append(["", text])
            continue

        for segment in speaker_ts_normalized:
            if end < segment[0]:
                break

            if start > segment[1]:
                continue

            suitable_segments.append(segment)
            suitable_speakers.append(segment[2])

        if not suitable_speakers:
            continue

        if len(suitable_speakers) == suitable_speakers.count(suitable_speakers[0]):
            speaker: str = str(suitable_speakers[0])

            if assigned and (speaker in assigned[-1][0]):
                assigned[-1][1] += " " + text
            else:
                assigned.append(["Speaker " + speaker, text])
            continue

        speaker_ranges: list = [0, 0] * 5  # ids for 10 speakers
        for segment in suitable_segments:
            speaker_ranges[segment[2]] += min(end, segment[1]) - max(start, segment[0])

        speaker: str = str(speaker_ranges.index(max(speaker_ranges)))
        if assigned and (speaker in assigned[-1][0]):
            assigned[-1][1] += " " + text
        else:
            assigned.append(["Speaker " + speaker, text])

    return assigned
