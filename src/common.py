import modal

# Model names + configs
PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"
WHISPERX_MODEL_NAME = "large-v2"

app = modal.App("asr-benchmark")
dataset_volume = modal.Volume.from_name("asr-benchmark-cache", create_if_missing=True)
model_cache = modal.Volume.from_name("asr-benchmark-data", create_if_missing=True)

GPUS = ["t4", "a10g", "h100"]

# Constants
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "ERROR": "\033[91m",
    "ENDC": "\033[0m",
}
