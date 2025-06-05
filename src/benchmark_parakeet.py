import modal
import json
import time
from typing import Any

from pathlib import Path

from src.common import app, dataset_volume, model_cache, GPUS
from src.utils import write_results

# NVIDIA GPU image is incompatible with CPU-only workloads.
parakeet_cpu_image = modal.Image.debian_slim(python_version="3.12")
parakeet_gpu_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
)

for image in [parakeet_cpu_image, parakeet_gpu_image]:
    image = (
        image.env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/cache",
                "DEBIAN_FRONTEND": "noninteractive",
                # "CXX": "g++",
                # "CC": "g++",
            }
        )
        .apt_install("ffmpeg")
        .pip_install(
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "nemo_toolkit[asr]==2.3.1",
            "cuda-python>=12.3",
        )
        .pip_install("numpy<2.0")  # Downgrade numpy; incompatible current version
        .entrypoint([])
        .add_local_python_source("src.common", "src.utils")
    )


MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=parakeet_cpu_image,
)
class ParakeetCPU:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import librosa
        import evaluate

        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.gpu = "cpu"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> dict[str, Any]:
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

        print("Time taken to transcribe: ", transcription_time)

        wer_model = evaluate.load("wer")
        actual = output[0].text
        wer = wer_model.compute(predictions=[actual], references=[expected])
        transcription = output[0].text

        return {
            "model": MODEL_NAME.replace("/", "-"),
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=parakeet_gpu_image,
)
class ParakeetGPU:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import librosa
        import evaluate

        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> dict[str, Any]:
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

        print("Time taken to transcribe: ", transcription_time)

        wer_model = evaluate.load("wer")
        actual = output[0].text
        wer = wer_model.compute(predictions=[actual], references=[expected])
        transcription = output[0].text

        return {
            "model": MODEL_NAME.replace("/", "-"),
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": self.gpu,
        }


@app.local_entrypoint()
def benchmark_parakeet():
    from pathlib import Path

    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    GPU_CLASSES = {
        "cpu": ParakeetCPU,
        **{gpu: ParakeetGPU.with_options(gpu=gpu) for gpu in GPUS},
    }

    for gpu, model_class in GPU_CLASSES:
        parakeet = model_class(gpu=gpu)

        results = list(parakeet.run.map(files))

        results_path = write_results(results, MODEL_NAME.replace("/", "-"))
        with dataset_volume.batch_upload() as batch:
            batch.put_file(results_path, f"/results/{results_path}")
