import modal
from src.common import app, dataset_volume, model_cache
from src.utils import write_results

MODEL_NAME = "openai/whisper-large-v3-turbo"

whisper_cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands(
        "uv pip install --system evaluate==0.4.3 jiwer==3.1.0 librosa==0.11.0 hf_transfer vllm[audio]"
    )
    .entrypoint([])
    .add_local_python_source("src.common", "src.utils")
)


with whisper_cpu_image.imports():
    import librosa
    from vllm import LLM, SamplingParams


whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("uv")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",
        }
    )
    .run_commands(
        "uv pip install --system evaluate==0.4.3 jiwer==3.1.0 librosa==0.11.0 hf_transfer vllm[audio]"
    )
    .entrypoint([])
    .add_local_python_source("src.common", "src.utils")
)


with whisper_image.imports():
    import librosa
    from vllm import LLM, SamplingParams


@app.cls(
    volumes={
        "/cache": model_cache,
        "/data": dataset_volume,
    },
    image=whisper_cpu_image,
)
class WhisperCPU:
    @modal.enter()
    def load(self):
        import json

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )
        self.gpu = "cpu"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str):
        import time
        from pathlib import Path
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

        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.time() - start_time

        if len(outputs) == 0:
            transcription = ""
            wer = None
        else:
            for output in outputs:
                transcription = output.outputs[0].text
                break
            wer_model = evaluate.load("wer")
            wer = wer_model.compute(predictions=[transcription], references=[expected])

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
    gpu="a10g",
    volumes={
        "/cache": model_cache,
        "/data": dataset_volume,
    },
    image=whisper_image,
)
class WhisperA10G:
    @modal.enter()
    def load(self):
        import json

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )
        self.gpu = "a10g"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str):
        import time
        from pathlib import Path
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

        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.time() - start_time

        if len(outputs) == 0:
            transcription = ""
            wer = None
        else:
            for output in outputs:
                transcription = output.outputs[0].text
                break
            wer_model = evaluate.load("wer")
            wer = wer_model.compute(predictions=[transcription], references=[expected])

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
    gpu="h100",
    volumes={
        "/cache": model_cache,
        "/data": dataset_volume,
    },
    image=whisper_image,
)
class WhisperH100:
    @modal.enter()
    def load(self):
        import json

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )
        self.gpu = "h100"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str):
        import time
        from pathlib import Path
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

        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.time() - start_time

        if len(outputs) == 0:
            transcription = ""
            wer = None
        else:
            for output in outputs:
                transcription = output.outputs[0].text
                break
            wer_model = evaluate.load("wer")
            wer = wer_model.compute(predictions=[transcription], references=[expected])

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
    gpu="t4",
    volumes={
        "/cache": model_cache,
        "/data": dataset_volume,
    },
    image=whisper_image,
)
class WhisperT4:
    @modal.enter()
    def load(self):
        import json

        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=448,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.95,
        )
        self.gpu = "t4"
        with open("/data/metadata.json", "r") as f:
            self.metadata = json.load(f)

    @modal.method()
    def run(self, file: str):
        import time
        from pathlib import Path
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

        outputs = self.llm.generate(prompts, sampling_params)
        transcription_time = time.time() - start_time

        if len(outputs) == 0:
            transcription = ""
            wer = None
        else:
            for output in outputs:
                transcription = output.outputs[0].text
                break
            wer_model = evaluate.load("wer")
            wer = wer_model.compute(predictions=[transcription], references=[expected])

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
def benchmark_whisper():
    from pathlib import Path

    files = [
        str(Path("/data") / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    GPU_CONFIG = {
        "a10g": WhisperA10G,
        "h100": WhisperH100,
        "t4": WhisperT4,
    }

    for _, model_class in GPU_CONFIG.items():
        whisper = model_class()
        results = list(whisper.run.map(files))
        results_path = write_results(results, MODEL_NAME.replace("/", "-"))
        with dataset_volume.batch_upload() as batch:
            batch.put_file(results_path, f"/results/{results_path}")
