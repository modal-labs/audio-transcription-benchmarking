# ---
# lambda-test: false
# ---

import csv
import json
from pathlib import Path

import modal
from common import app, dataset_volume

image = (
    modal.Image.debian_slim().pip_install("pandas").add_local_python_source("common")
)


@app.function(
    volumes={"/data": dataset_volume},
    image=image,
)
def upload_token_counts():
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
