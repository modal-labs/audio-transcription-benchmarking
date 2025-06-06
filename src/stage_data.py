from pathlib import Path

import modal
from src.common import app, dataset_volume, DATASET_VOLUME_NAME

import csv
import json


DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

base_image = modal.Image.debian_slim()

download_image = base_image.pip_install("requests==2.32.3")
data_prep_image = base_image.pip_install("numpy==2.2.6")
metadata_image = base_image.pip_install("pandas==2.3.0")


def _build_image(image):
    return image.add_local_python_source("src.common", "src.utils")


download_image = _build_image(download_image)
data_prep_image = _build_image(data_prep_image)
metadata_image = _build_image(metadata_image)


@app.function(
    volumes={"/data": dataset_volume},
    image=download_image,
    timeout=1200,  # 20 minutes
)
def download_lj_data():
    import tarfile
    import tempfile

    import requests

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tar_path = tmpdir_path / "LJSpeech-1.1.tar.bz2"

        print("📥 Downloading dataset...")
        with requests.get(DATA_URL, stream=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("📦 Extracting dataset...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=tmpdir_path)

        dataset_dir = tmpdir_path / "LJSpeech-1.1"

        print("☁️ Uploading to Modal volume under 'raw/'...")
        file_count = 0
        with dataset_volume.batch_upload() as batch:
            for path in dataset_dir.rglob("*"):
                if path.is_file():
                    relative_path = path.relative_to(dataset_dir)
                    remote_path = f"/raw/{relative_path}"
                    batch.put_file(str(path), remote_path)
                    file_count += 1

        print(f"✅ Uploaded {file_count} files to Modal volume {DATASET_VOLUME_NAME}")


@app.function(
    volumes={"/data": dataset_volume},
    image=data_prep_image,
)
def convert_to_mono_16khz(input_path: str, output_dir: str):
    """Converts an input WAV file to 16khz mono and stores output in `output_path` WAV file."""
    import wave
    from pathlib import Path

    import numpy as np

    # Open the input WAV file
    filename = Path(input_path).name
    input_path = str(Path("/data") / input_path)
    with wave.open(input_path, "rb") as wav_in:
        # Get the input parameters
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()

        # Read all frames
        frames = wav_in.readframes(n_frames)

    # Convert frames to numpy array
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Reshape the array based on number of channels
    audio_data = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels)
        # Convert to mono by averaging all channels
        audio_data = audio_data.mean(axis=1).astype(dtype)

    # Resample to 16 kHz if needed
    if frame_rate != 16000:
        # Calculate resampling ratio
        ratio = 16000 / frame_rate
        new_length = int(len(audio_data) * ratio)

        # Resample using linear interpolation
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(
            dtype
        )

    # Create a new WAV file
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_path = str(Path(output_dir) / filename)
    with wave.open(output_path, "wb") as wav_out:
        wav_out.setnchannels(1)  # Mono
        wav_out.setsampwidth(sample_width)
        wav_out.setframerate(16000)  # 16 kHz
        wav_out.writeframes(audio_data.tobytes())


@app.function(
    volumes={"/data": dataset_volume},
    image=data_prep_image,
)
def preprocess_wav_files():
    import itertools

    print("Processing files to 16kHz mono...")
    output_dir = "/data/processed"
    files = [
        f.path for f in dataset_volume.listdir("/raw/wavs") if f.path.endswith(".wav")
    ]
    inputs = list(zip(files, itertools.cycle([output_dir])))

    # Process all files in parallel
    list(convert_to_mono_16khz.starmap(inputs))
    print(
        f"✨ All done! Data processed and available in the {output_dir} in the Modal Volume {DATASET_VOLUME_NAME}"
    )


@app.function(
    volumes={"/data": dataset_volume},
    image=metadata_image,
)
def parse_metadata():
    metadata_path = Path("/data/raw/metadata.csv")
    output_path = Path("/data/metadata.json")

    file_data = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 3:
                continue
            wav_filename = f"{row[0]}.wav"
            normalized_transcription = row[2]
            token_count = len(normalized_transcription.strip().split())
            file_data[wav_filename] = {
                "token_count": token_count,
                "transcription": normalized_transcription,
            }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2)

    print(f"✅ Wrote metadata.json with {len(file_data)} entries.")


@app.function(
    volumes={"/data": dataset_volume},
    image=data_prep_image,
    timeout=1200,  # 20 minutes
)
def stage_data():
    download_and_upload_lj_data.remote()
    preprocess_wav_files.remote()
    parse_metadata.remote()
