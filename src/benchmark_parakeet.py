import modal
import json
import time
from pathlib import Path

from src.common import (
    app,
    dataset_volume,
    model_cache,
    GPUS,
    MODEL_CACHE_PATH,
    PARAKEET_MODEL_DISPLAY_NAME,
    PARAKEET_MODEL_NAME,
    METADATA_PATH,
    BenchmarkResult,
    DATASET_PATH,
)
from src.utils import write_results


# NVIDIA GPU image is incompatible with CPU-only workloads.
parakeet_cpu_image = modal.Image.debian_slim(python_version="3.12")
parakeet_gpu_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
)


def _build_image(image):
    return (
        image.env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/cache",
                "DEBIAN_FRONTEND": "noninteractive",
                "CXX": "g++",
                "CC": "g++",
            }
        )
        .apt_install("ffmpeg")
        .pip_install(
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "cuda-python>=12.3",
            "nemo_toolkit[asr]==2.3.1",
        )
        .pip_install("numpy<2.0")  # Downgrade numpy; incompatible current version
        .entrypoint([])
        .add_local_python_source("src.common", "src.utils")
    )


parakeet_cpu_image = _build_image(parakeet_cpu_image)
parakeet_gpu_image = _build_image(parakeet_gpu_image)


@app.cls(
    volumes={
        "/data": dataset_volume,
        MODEL_CACHE_PATH.as_posix(): model_cache,
    },
    image=parakeet_cpu_image,
)
class ParakeetCPU:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=PARAKEET_MODEL_NAME
        )
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: Path) -> BenchmarkResult:
        import librosa
        import evaluate

        benchmark_result = BenchmarkResult(
            model=PARAKEET_MODEL_DISPLAY_NAME,
            filename=file.name,
            gpu=self.gpu,
        )

        filename = file.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return benchmark_result

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file, sr=None)
        duration = len(y) / float(sr)

        start_time = time.perf_counter()
        output = self.model.transcribe([str(file)])
        transcription_time = time.perf_counter() - start_time

        print("Transcription time: ", transcription_time)

        wer_model = evaluate.load("wer")
        actual = output[0].text
        wer = wer_model.compute(predictions=[actual], references=[expected])
        transcription = output[0].text

        benchmark_result.expected_transcription = expected
        benchmark_result.transcription = transcription
        benchmark_result.transcription_time = transcription_time
        benchmark_result.audio_duration = duration
        benchmark_result.wer = wer
        return benchmark_result


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
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=PARAKEET_MODEL_NAME
        )
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: Path) -> BenchmarkResult:
        import librosa
        import evaluate

        benchmark_result = BenchmarkResult(
            model=PARAKEET_MODEL_DISPLAY_NAME,
            filename=file.name,
            gpu=self.gpu,
        )

        filename = file.name

        file_metadata = self.metadata.get(filename)
        if file_metadata is None:
            return benchmark_result

        expected = file_metadata["transcription"]

        y, sr = librosa.load(file, sr=None)
        duration = len(y) / float(sr)

        start_time = time.perf_counter()
        output = self.model.transcribe([str(file)])
        transcription_time = time.perf_counter() - start_time

        print("Transcription time: ", transcription_time)

        wer_model = evaluate.load("wer")
        actual = output[0].text
        wer = wer_model.compute(predictions=[actual], references=[expected])
        transcription = output[0].text

        benchmark_result.expected_transcription = expected
        benchmark_result.transcription = transcription
        benchmark_result.transcription_time = transcription_time
        benchmark_result.audio_duration = duration
        benchmark_result.wer = wer
        return benchmark_result


@app.function(
    volumes={
        MODEL_CACHE_PATH.as_posix(): dataset_volume,
    },
    image=parakeet_gpu_image,
)
def benchmark_parakeet():
    files = [
        (DATASET_PATH / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    GPU_CLASSES = {
        "cpu": ParakeetCPU,
        **{gpu: ParakeetGPU.with_options(gpu=gpu) for gpu in GPUS},
    }

    for gpu, model_class in GPU_CLASSES.items():
        parakeet = model_class(gpu=gpu)
        results = list(parakeet.run.map(files))
        write_results.remote(results, PARAKEET_MODEL_DISPLAY_NAME)