import json
import time
from pathlib import Path
from typing import Any

from src.common import COLOR


def write_results(results: list[dict[str, Any]], model_name: str):
    """Write JSONL dataset with all results."""
    timestamp = int(time.time())
    result_path = Path(f"result_{model_name}_{timestamp}.jsonl")
    with result_path.open("w") as f:
        for result in results:
            if result and not result.get("expected_transcription"):
                continue
            f.write(json.dumps(result) + "\n")

    return result_path


def print_header(text):
    print(f"{COLOR['HEADER']}{text}{COLOR['ENDC']}")


def print_error(text):
    print(f"{COLOR['ERROR']}{text}{COLOR['ENDC']}")
