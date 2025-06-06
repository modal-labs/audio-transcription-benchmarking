# ## Benchmarking Audio-to-Text Models - Parakeet, Whisper and WhisperX

# This example demonstrates how to benchmark multiple audio-to-text models at
# lightning fast speeds. In just a single Modal app, we will:
# 1. Download and upload the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) to a Modal Volume.
# 2. Process the audio `.wav` files into a suitable format.
# 3. Benchmark the Parakeet, Whisper and WhisperX models in parallel.
# 4. Postprocess and plot the results, and save them to a Modal volume.

# The full code can be found [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/audio-to-text/benchmarking).
# We'll start by importing the necessary modules and defining the model configurations.

import asyncio
from itertools import product

from src.benchmark_parakeet import ParakeetCPU, ParakeetGPU
from src.benchmark_whisper import Whisper
from src.benchmark_whisperx import WhisperX
from src.common import (
    PARAKEET_MODEL_DISPLAY_NAME,
    WHISPER_MODEL_DISPLAY_NAME,
    WHISPERX_MODEL_DISPLAY_NAME,
    app,
    dataset_volume,
    GPUS,
    DATASET_PATH,
)
from src.stage_data import (
    stage_data,
)
from src.postprocess_results import postprocess_results
from src.utils import print_error, print_header, write_results

MODEL_CONFIGS = [
    (PARAKEET_MODEL_DISPLAY_NAME, ParakeetGPU),
    (WHISPER_MODEL_DISPLAY_NAME, Whisper),
    (WHISPERX_MODEL_DISPLAY_NAME, WhisperX),
]

# ## Download and upload data

# The full download, extracting and uploading takes about 9 minutes.
#
# To skip download (on subsequent runs), set `REDOWNLOAD_DATA` to False.
REDOWNLOAD_DATA = False


# ## Run model inference in parallel
# We'll run the models in parallel using `asyncio`, and then postprocess the results.
# This is super fast! We use `.map` to offload the expensive computation, spinning
# up multiple containers in parallel.


@app.function(
    volumes={
        "/data": dataset_volume,
    },
)
def run_model_sync(model_name, instance, gpu, files):
    import json

    results = list(instance.with_options(gpu=gpu)().run.map(files))
    results_path = write_results(results, model_name)
    with open(results_path, "w") as f:
        f.write(json.dumps(results))
    print(f"✅ {model_name} results uploaded to /results/{results_path}")
    return model_name, results


def check_data_exists():
    from grpclib import GRPCError
    from grpclib.const import Status

    try:
        dataset_volume.listdir("/raw/wavs")
        dataset_volume.listdir("/processed")
    except GRPCError as e:
        if e.status == Status.NOT_FOUND:
            print_error(
                "Data not found in volume. Please re-run app.py with REDOWNLOAD_DATA=True. Note that this will take several minutes.",
            )
            exit(1)


@app.local_entrypoint()
async def main():
    from pathlib import Path

    if REDOWNLOAD_DATA:
        stage_data.remote()
    else:
        print("Skipping data download")
        check_data_exists()

    print_header("⚡️ Benchmarking all models in parallel...")
    files = [
        (DATASET_PATH / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ][:1]

    print(f"Found {len(files)} files to benchmark")

    model_parameters = [(*mc, gc) for mc, gc in product(MODEL_CONFIGS, GPUS)] + [
        (PARAKEET_MODEL_DISPLAY_NAME, ParakeetCPU, "cpu")
    ]
    tasks = [
        asyncio.get_event_loop().run_in_executor(
            None, run_model_sync, model_name, instance, gpu, files
        )
        for model_name, instance, gpu in model_parameters
    ]
    await asyncio.gather(*tasks)

    print_header("🔮 Postprocessing results...")
    postprocess_results.remote()


# ## Plotting the results
# The results are saved to the same Modal Volume as the data, in the `/data/analysis`
# directory!
