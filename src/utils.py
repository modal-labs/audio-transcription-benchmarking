import json
import time
from pathlib import Path

from common import COLOR


def write_results(results: dict, model_name: str):
    timestamp = int(time.time())
    result_path = Path(f"result_{model_name}_{timestamp}.jsonl")
    with result_path.open("w") as f:
        for result in results:
            if result["expected_transcription"] is None:
                continue
            f.write(json.dumps(result) + "\n")
    return result_path


def print_header(text):
    print(f"{COLOR['HEADER']}{text}{COLOR['ENDC']}")


def print_error(text):
    print(f"{COLOR['ERROR']}{text}{COLOR['ENDC']}")
