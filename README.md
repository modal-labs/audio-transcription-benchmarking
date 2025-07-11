# Audio Transcription Benchmarking

This folder contains code that compares different ASR models using
Modal. Models are compared against a sample dataset of WAV files. Available
models:

- [**nvidia/parakeet-tdt-0.6b-v2**](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2): using NeMo as inference engine
- [**whisper-large-v3-turbo**](https://huggingface.co/openai/whisper-large-v3-turbo): using vLLM
- [**whisperx**](https://github.com/m-bain/whisperX): using Python

## Benchmarks

- Word Error Rate (WER)
- Throughput (RTFx)

## Local Development

### Modal

- Install Modal in your current Python virtual environment (`pip install modal`)
- Set up your Modal account: `python3 -m modal setup`

### Create secret (WhisperX only)

First, create a [Hugging Face access token](https://huggingface.co/settings/tokens) and store it in a [Modal Secret](https://modal.com/docs/guide/secrets#secrets) called `huggingface-token`.
We'll use this secret later to download models from the Huggingface Hub. This is only required for our WhisperX model - you can skip this step if you only
want to run Whisper or Parakeet.

### Prepare dataset

We'll use the publicly available [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) to benchmark our models.

Run the following script to upload all data to a new [Modal Volume](https://modal.com/docs/guide/volumes#volumes)
and convert all WAV files to equivalent files but in 16khz and mono. This makes files compatible with Parakeet. The data in this volume can be accessed by all apps.

```shell
modal run -m src.stage_data
```

### Inference

You can benchmark a model by `modal run`ning either of the files prefixed by `benchmark_`, such as

```
modal run -m src.benchmark_parakeet
```

Modal will scale to add as many GPUs as necessary in order to process your
dataset. Outputs will be available in a local CSV file named:

```shell
result_parakeet_$TIMESTAMP.csv
```

## Post-Process Results

When you're ready, move all your final results JSONL files to `results/`. Replace the filename below with your selected files:

```shell
mkdir -p results
mv result_nvidia-parakeet-tdt-0.6b-v2_123456789.jsonl results
```

Generate visualizations in the Jupyter Notebook:

```shell
pip install -r requirements/requirements-dev.txt
jupyter notebook
```
