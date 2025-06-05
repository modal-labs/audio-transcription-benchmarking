from pathlib import Path

import modal
from src.common import DATASET_VOLUME_NAME, app, dataset_volume

# Full dataset
DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

# Subset of dataset
LOCAL_ZIP_PATH = Path(__file__).parent / "LJSpeech-1.1-subset.zip"

image = (
    modal.Image.debian_slim()
    .pip_install("requests==2.32.3")
    .add_local_python_source("src.common", "src.utils")
)


@app.function(
    volumes={"/data": dataset_volume},
    image=image,
    timeout=1200,  # 20 minutes
)
def download_and_upload_lj_data():
    import tarfile
    import tempfile
    from pathlib import Path

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
