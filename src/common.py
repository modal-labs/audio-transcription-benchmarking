import modal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Model names + configs
PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"
WHISPERX_MODEL_NAME = "large-v2"

PARAKEET_MODEL_DISPLAY_NAME = PARAKEET_MODEL_NAME.replace("/", "-")
WHISPER_MODEL_DISPLAY_NAME = WHISPER_MODEL_NAME.replace("/", "-")
WHISPERX_MODEL_DISPLAY_NAME = f"whisperx-{WHISPERX_MODEL_NAME}"

# DATASET_VOLUME_NAME = "asr-benchmark-cache"
# MODEL_CACHE_VOLUME_NAME = "asr-benchmark-data"

MODEL_CACHE_PATH = Path("/cache")
MODEL_CACHE_VOLUME_NAME = "audio-diarization-model-cache"
DATASET_PATH = Path("/data")
DATASET_VOLUME_NAME = "audio-diarization-benchmarking-data"

METADATA_PATH = Path("/data/metadata.json")

app = modal.App("asr-benchmark")
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)

GPUS = ["t4", "a10g", "h100"]

# Constants
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "ERROR": "\033[91m",
    "ENDC": "\033[0m",
}


@dataclass
class BenchmarkResult:
    model: str
    filename: str
    gpu: str
    expected_transcription: Optional[str] = None
    transcription: Optional[str] = None
    transcription_time: Optional[float] = None
    audio_duration: Optional[float] = None
    wer: Optional[float] = None
