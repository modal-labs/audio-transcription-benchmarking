import modal
from src.common import app, dataset_volume, model_cache
from src.utils import write_results

parakeet_cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
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
    .add_local_python_source("src.common", "src.utils")
)

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
    .add_local_python_source("src.common", "src.utils")
)

with parakeet_cpu_image.imports():
    import nemo.collections.asr as nemo_asr


with parakeet_image.imports():
    import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=parakeet_cpu_image,
)
class ParakeetCPU:
    @modal.enter()
    def load(self):
        import json

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.gpu = "cpu"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float, float]:
        import time
        from pathlib import Path

        import librosa
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

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
    gpu="a10g",
    image=parakeet_image,
)
class ParakeetA10G:
    @modal.enter()
    def load(self):
        import json

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.gpu = "a10g"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float, float]:
        import time
        from pathlib import Path

        import librosa
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

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
    gpu="h100",
    image=parakeet_image,
)
class ParakeetH100:
    @modal.enter()
    def load(self):
        import json

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.gpu = "h100"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float, float]:
        import time
        from pathlib import Path

        import librosa
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

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
    gpu="t4",
    image=parakeet_image,
)
class ParakeetT4:
    @modal.enter()
    def load(self):
        import json

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        self.gpu = "t4"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str) -> tuple[str, str, float, float, float]:
        import time
        from pathlib import Path

        import librosa
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

        # Time the transcription
        start_time = time.time()
        output = self.model.transcribe([file])
        transcription_time = time.time() - start_time

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
    GPU_CONFIG = {
        "cpu": ParakeetCPU,
        "a10g": ParakeetA10G,
        "h100": ParakeetH100,
        "t4": ParakeetT4,
    }

    for _, model_class in GPU_CONFIG.items():
        parakeet = model_class()

        results = list(parakeet.run.map(files))

        results_path = write_results(results, MODEL_NAME.replace("/", "-"))
        with dataset_volume.batch_upload() as batch:
            batch.put_file(results_path, f"/results/{results_path}")
