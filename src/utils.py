import json
import time
from pathlib import Path
import dataclasses
from src.common import COLOR, BenchmarkResult, RESULTS_PATH, app, dataset_volume


@app.function(
    volumes={
        "/data": dataset_volume,
    },
    timeout=60 * 60, # 1 hour
)
def write_results(results: list[BenchmarkResult], model_name: str):
    """Write JSONL dataset with all results."""
    timestamp = int(time.time())
    result_path = RESULTS_PATH / Path(f"result_{model_name}_{timestamp}.jsonl")
    with open(result_path, "w+") as f:
        for result in results:
            if result.expected_transcription is None:
                continue
            dct = dataclasses.asdict(result)
            f.write(json.dumps(dct) + "\n")
    print(f"Wrote results to {result_path}")


def print_header(text):
    print(f"{COLOR['HEADER']}{text}{COLOR['ENDC']}")


def print_error(text):
    print(f"{COLOR['ERROR']}{text}{COLOR['ENDC']}")
