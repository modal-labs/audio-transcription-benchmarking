# ---
# lambda-test: false
# ---

import modal
from common import app, dataset_volume, model_cache
from utils import write_results

parakeet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .run_commands(
        "uv pip install --system evaluate==0.4.3 librosa==0.11.0 hf_transfer huggingface_hub[hf-xet] nemo_toolkit[asr] cuda-python>=12.3",
        "uv pip install --system 'numpy<2.0'",  # downgrade numpy; incompatible current version
    )
    .entrypoint([])
    .add_local_python_source("common", "utils")
)


with parakeet_image.imports():
    import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
GPU = "a10g"
GPU_TYPES = ["t4", "a10g", "h100"]


image = (
    modal.Image.debian_slim().pip_install("pandas").add_local_python_source("common")
)


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    gpu="a10g",
    image=parakeet_image,
)
class ParakeetA10G:
    @modal.enter()
    def load(self):
        import json

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float, float]:
        import time
        from pathlib import Path

        import librosa
        import evaluate

        file_path = Path(file)

        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / float(sr)

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

        wer = evaluate.load("wer")
        filename = file.split("/")[-1]
        expected = self.metadata[filename]["transcription"]
        actual = output[0].text
        wer = wer.compute(predictions=[actual], references=[expected])
        print("WER", wer)
        print("Expected", expected)
        print("Actual", actual)

        transcription = output[0].text

        return {
            "model": MODEL_NAME.replace("/", "-"),
            "filename": filename,
            "expected_transcription": expected,
            "transcription": transcription,
            "transcription_time": transcription_time,
            "audio_duration": duration,
            "wer": wer,
            "gpu": "a10g",
        }


@app.local_entrypoint()
def benchmark_parakeet():
    from pathlib import Path

    GPU_CONFIG = {
        "a10g": ParakeetA10G,
    }

    for _, model_class in GPU_CONFIG.items():
        parakeet = model_class()

        # Convert paths to strings for serialization
        files = [
            str(Path("/data") / Path(f.path))
            for f in dataset_volume.listdir("/processed")
        ]

        results = list(parakeet.run.map(files))

        results_path = write_results(results, MODEL_NAME.replace("/", "-"))
        with dataset_volume.batch_upload() as batch:
            batch.put_file(results_path, f"/results/{results_path}")
