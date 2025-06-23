import time
from pathlib import Path

import modal
from src.common import (
    app,
    dataset_volume,
    model_cache,
    GPUS,
    BenchmarkResult,
    MODEL_CACHE_PATH,
    WHISPERX_MODEL_NAME,
    WHISPERX_MODEL_DISPLAY_NAME,
    METADATA_PATH,
    DATASET_PATH,
)
from src.utils import write_results

whisperx_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_CACHE_PATH.as_posix(),
        }
    )
    .apt_install("ffmpeg")
    .pip_install(
        "evaluate==0.4.3",
        "jiwer==3.1.0",
        "librosa==0.11.0",
        "hf_transfer==0.1.9",
        "faster-whisper==1.1.1",
        "whisperx==3.3.4",
        "torchaudio==2.7.1",
    )
    .entrypoint([])
    .add_local_python_source("src.common", "src.utils")
)


with whisperx_image.imports():
    import librosa
    import whisperx
    import evaluate
    from pathlib import Path


@app.cls(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        DATASET_PATH.as_posix(): dataset_volume,
        MODEL_CACHE_PATH.as_posix(): model_cache,
    },
    image=whisperx_image,
)
class WhisperX:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import json

        device = "cuda"
        self.model = whisperx.load_model(
            WHISPERX_MODEL_NAME, device, compute_type="float16"
        )
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: Path) -> BenchmarkResult:
        benchmark_result = BenchmarkResult(
            model=WHISPERX_MODEL_DISPLAY_NAME,
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
        audio = whisperx.load_audio(file, sr=16000)
        result = self.model.transcribe(audio, batch_size=16)
        transcription_time = time.perf_counter() - start_time

        print("Transcription time: ", transcription_time)

        transcription = " ".join([s["text"] for s in result["segments"]])

        wer_model = evaluate.load("wer")
        wer = wer_model.compute(predictions=[transcription], references=[expected])

        benchmark_result.expected_transcription = expected
        benchmark_result.transcription = transcription
        benchmark_result.transcription_time = transcription_time
        benchmark_result.audio_duration = duration
        benchmark_result.wer = wer
        return benchmark_result


@app.function(
    volumes={
        DATASET_PATH.as_posix(): dataset_volume,
        MODEL_CACHE_PATH.as_posix(): model_cache,
    },
    image=whisperx_image,
)
def benchmark_whisperx():
    files = [
        (DATASET_PATH / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]
    for gpu in GPUS:
        whisperx = WhisperX.with_options(gpu=gpu)(gpu=gpu)
        results = list(whisperx.run.map(files))
        write_results.remote(results, WHISPERX_MODEL_DISPLAY_NAME)
