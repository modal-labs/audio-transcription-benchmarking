import modal
from src.common import app, dataset_volume, model_cache

image = (
    modal.Image.debian_slim()
    .pip_install("librosa", "soundfile", "numpy")
    .apt_install("ffmpeg")
    .add_local_python_source("src.common")
)


@app.function(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=image,
    timeout=3600,
)
def stitch_audio_files(files: list[str], output_file: str):
    from pathlib import Path
    import librosa
    import numpy as np
    import soundfile as sf

    input_dir = "/data/processed"
    output_dir = "/data/processed_long"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir) / output_file

    input_path = Path(input_dir)

    file_set = set([f"{f}.wav" for f in files])
    potential_files = input_path.glob(f"{files}*.wav")
    wav_files = sorted([f for f in potential_files if f.name in file_set])

    if not wav_files:
        print("No .wav files found.")
        return

    all_audio = []
    sample_rate = None

    for file in wav_files:
        print(f"Loading {file.name}")
        y, sr = librosa.load(file, sr=None)  # Keep original sample rate
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch in {file.name}: {sr} != {sample_rate}"
            )
        all_audio.append(y)

    stitched_audio = np.concatenate(all_audio)
    sf.write(output_file, stitched_audio, sample_rate)
    print(f"Exported stitched file to {output_file}")


@app.function(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=image,
)
def main():
    import itertools

    files = []
    transcriptions = []
    with open("/data/metadata.csv", "r") as f:
        for line in f:
            files.append(line.split("|")[0])
            transcriptions.append(line.split("|")[2])

    # Split files into chunks of 1000
    CHUNK_SIZE = 50
    chunks = []
    for i in range(0, len(files), CHUNK_SIZE):
        chunks.append(files[i : i + CHUNK_SIZE])

    print(f"Stitching {len(chunks)} chunks")

    #### Temporarily limiting this to 1 chunk
    inputs = list(
        zip(chunks, itertools.cycle([f"chunk_{i}.wav" for i in range(len(chunks))]))
    )[:1]
    list(stitch_audio_files.starmap(inputs))
