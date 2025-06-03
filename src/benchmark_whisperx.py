# ---
# lambda-test: false  # missing-secret
# ---

import time
from pathlib import Path

import modal
from common import app, dataset_volume, model_cache
from utils import write_results

MODEL_NAME = "large-v2"


whisperx_cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands(
        "uv pip install --system evaluate==0.4.3 jiwer==3.1.0 librosa==0.11.0 hf_transfer faster-whisper whisperx torchaudio"
    )
    .apt_install("ffmpeg")
    .entrypoint([])
    .add_local_python_source("common", "utils")
)

with whisperx_cpu_image.imports():
    import librosa
    import whisperx


whisperx_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands(
        "uv pip install --system evaluate==0.4.3 jiwer==3.1.0 librosa==0.11.0 hf_transfer faster-whisper whisperx torchaudio"
    )
    .apt_install("ffmpeg")
    .entrypoint([])
    .add_local_python_source("common", "utils")
)


with whisperx_image.imports():
    import librosa
    import whisperx


@app.cls(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=whisperx_cpu_image,
)
class WhisperXCPU:
    @modal.enter()
    def load(self):
        import json

        device = "cpu"
        self.model = whisperx.load_model(MODEL_NAME, device, compute_type="float16")
        self.gpu = "cpu"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        import evaluate

        file_path = Path(file)
        filename = file_path.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return {
                "model": MODEL_NAME.replace("/", "-"),
                "filename": filename,
                "expected_transcription": None,
                "transcription": None,
                "transcription_time": None,
                "audio_duration": None,
                "wer": None,
                "gpu": self.gpu,
            }

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        start_time = time.time()
        audio = whisperx.load_audio(file_path, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.time() - start_time

        transcription = " ".join([s["text"] for s in result["segments"]])

        wer_model = evaluate.load("wer")
        wer = wer_model.compute(predictions=[transcription], references=[expected])

        return {
            "model": f"whisperx-{MODEL_NAME}",
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.cls(
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=whisperx_image,
)
class WhisperXA10G:
    @modal.enter()
    def load(self):
        import json

        device = "cuda"
        self.model = whisperx.load_model(MODEL_NAME, device, compute_type="float16")
        self.gpu = "a10g"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        import evaluate

        file_path = Path(file)
        filename = file_path.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return {
                "model": MODEL_NAME.replace("/", "-"),
                "filename": filename,
                "expected_transcription": None,
                "transcription": None,
                "transcription_time": None,
                "audio_duration": None,
                "wer": None,
                "gpu": self.gpu,
            }

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        start_time = time.time()
        audio = whisperx.load_audio(file_path, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.time() - start_time

        transcription = " ".join([s["text"] for s in result["segments"]])

        wer_model = evaluate.load("wer")
        wer = wer_model.compute(predictions=[transcription], references=[expected])

        return {
            "model": f"whisperx-{MODEL_NAME}",
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.cls(
    gpu="h100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=whisperx_image,
)
class WhisperXH100:
    @modal.enter()
    def load(self):
        import json

        device = "cuda"
        self.model = whisperx.load_model(MODEL_NAME, device, compute_type="float16")
        self.gpu = "h100"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        import evaluate

        file_path = Path(file)
        filename = file_path.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return {
                "model": MODEL_NAME.replace("/", "-"),
                "filename": filename,
                "expected_transcription": None,
                "transcription": None,
                "transcription_time": None,
                "audio_duration": None,
                "wer": None,
                "gpu": self.gpu,
            }

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        start_time = time.time()
        audio = whisperx.load_audio(file_path, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.time() - start_time

        transcription = " ".join([s["text"] for s in result["segments"]])

        wer_model = evaluate.load("wer")
        wer = wer_model.compute(predictions=[transcription], references=[expected])

        return {
            "model": f"whisperx-{MODEL_NAME}",
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.cls(
    gpu="t4",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=whisperx_image,
)
class WhisperXT4:
    @modal.enter()
    def load(self):
        import json

        device = "cuda"
        self.model = whisperx.load_model(MODEL_NAME, device, compute_type="float16")
        self.gpu = "t4"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float]:
        import evaluate

        file_path = Path(file)
        filename = file_path.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return {
                "model": MODEL_NAME.replace("/", "-"),
                "filename": filename,
                "expected_transcription": None,
                "transcription": None,
                "transcription_time": None,
                "audio_duration": None,
                "wer": None,
                "gpu": self.gpu,
            }

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        start_time = time.time()
        audio = whisperx.load_audio(file_path, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.time() - start_time

        transcription = " ".join([s["text"] for s in result["segments"]])

        wer_model = evaluate.load("wer")
        wer = wer_model.compute(predictions=[transcription], references=[expected])

        return {
            "model": f"whisperx-{MODEL_NAME}",
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.local_entrypoint()
def benchmark_whisperx():
    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ][:1]
    GPU_CONFIG = {
        "cpu": WhisperXCPU,
        # "a10g": WhisperXA10G,
        # "h100": WhisperXH100,
        # "t4": WhisperXT4,
    }

    for _, model_class in GPU_CONFIG.items():
        whisperx = model_class()
        results = list(whisperx.run.map(files))
        results_path = write_results(results, f"whisperx-{MODEL_NAME}")
        with dataset_volume.batch_upload() as batch:
            batch.put_file(results_path, f"/results/{results_path}")
