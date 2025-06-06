import modal
from src.common import (
    app,
    dataset_volume,
    model_cache,
    GPUS,
    MODEL_CACHE_PATH,
    BenchmarkResult,
    WHISPER_MODEL_DISPLAY_NAME,
    WHISPER_MODEL_NAME,
    METADATA_PATH,
    DATASET_PATH,
)
from src.utils import write_results
from pathlib import Path


whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "evaluate==0.4.3",
        "jiwer==3.1.0",
        "librosa==0.11.0",
        "hf_transfer==0.1.9",
        "vllm[audio]==0.9.0.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_CACHE_PATH.as_posix(),
        }
    )
    .entrypoint([])
    .add_local_python_source("src.common", "src.utils")
)


with whisper_image.imports():
    import librosa
    from vllm import LLM, SamplingParams
    import evaluate
    from pathlib import Path
    import time


@app.cls(
    volumes={
        DATASET_PATH.as_posix(): dataset_volume,
        MODEL_CACHE_PATH.as_posix(): model_cache,
    },
    image=whisper_image,
)
class Whisper:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import json

        self.llm = LLM(
            model=WHISPER_MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )
        with open(METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: Path) -> BenchmarkResult:
        benchmark_result = BenchmarkResult(
            model=WHISPER_MODEL_DISPLAY_NAME,
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

        prompts = [
            {
                "prompt": "<|startoftranscript|>",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            }
        ]

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        start_time = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.perf_counter() - start_time

        print("Transcription time: ", transcription_time)

        if len(outputs) == 0:
            transcription = ""
            wer = None
        else:
            for output in outputs:
                transcription = output.outputs[0].text
                break
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
    image=whisper_image,
    timeout=1200,
)
def benchmark_whisper():
    from pathlib import Path

    files = [
        (DATASET_PATH / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ][:1]

    for gpu in GPUS:
        whisper = Whisper.with_options(gpu=gpu)(gpu=gpu)
        results = list(whisper.run.map(files))
        write_results.remote(results, WHISPER_MODEL_DISPLAY_NAME)
