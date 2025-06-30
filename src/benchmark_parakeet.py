import modal
import json
import time
from pathlib import Path
from typing import List, Dict, NamedTuple

from src.common import (
    app,
    dataset_volume,
    model_cache,
    # GPUS,
    MODEL_CACHE_PATH,
    PARAKEET_MODEL_DISPLAY_NAME,
    PARAKEET_MODEL_NAME,
    METADATA_PATH,
    BenchmarkResult,
    DATASET_PATH,
)
from src.utils import write_results

GPUS = ["h200"]
BATCH_SIZE = 128
BATCHES_PER_CALL = round(11822/BATCH_SIZE + 1)


class TranscriptionResult(NamedTuple):
    filename: str
    transcription: str
    transcription_time: float


# NVIDIA GPU image is incompatible with CPU-only workloads.
parakeet_cpu_image = modal.Image.debian_slim(python_version="3.12")
parakeet_gpu_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
)


def _build_image(image):
    return (
        image.env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/cache",
                # "DEBIAN_FRONTEND": "noninteractive",
                "CXX": "g++",
                "CC": "g++",
            }
        )
        .apt_install("ffmpeg")
        .pip_install(
            "torch",
            "evaluate==0.4.3",
            "librosa==0.11.0",
            "hf_transfer==0.1.9",
            "huggingface_hub[hf-xet]==0.32.4",
            "cuda-python==12.8.0",
            "nemo_toolkit[asr]==2.3.1",
            
        )
        # .pip_install("numpy<2.0")  # Downgrade numpy; incompatible current version
        .entrypoint([])
        .add_local_python_source("src.common", "src.utils")
    )


parakeet_cpu_image = _build_image(parakeet_cpu_image)
parakeet_gpu_image = _build_image(parakeet_gpu_image)


@app.cls(
    volumes={
        "/data": dataset_volume,
        MODEL_CACHE_PATH.as_posix(): model_cache,
    },
    image=parakeet_cpu_image,
)
class ParakeetCPU:
    gpu: str = modal.parameter()

    @modal.enter()
    def load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=PARAKEET_MODEL_NAME
        )

    @modal.method()
    def run(self, files: List[Path]) -> List[TranscriptionResult]:
        # Batch transcription
        file_paths = [str(file) for file in files]
        start_time = time.perf_counter()
        outputs = self.model.transcribe(file_paths)
        transcription_time = time.perf_counter() - start_time

        print(f"Batch transcription time for {len(files)} files: {transcription_time}")

        # Return transcription results
        results = []
        for i, file in enumerate(files):
            results.append(TranscriptionResult(
                filename=file.name,
                transcription=outputs[i].text,
                transcription_time=transcription_time / len(files)  # Average time per file
            ))

        return results


@app.cls(
    volumes={
        "/data": dataset_volume,
        "/cache": model_cache,
    },
    image=parakeet_gpu_image,
    max_containers=1,
    # region="us-east-1",
    enable_memory_snapshot=True,
    timeout=60*60,
)
# @modal.concurrent(max_inputs=6,target_inputs=4)
class ParakeetGPU:
    gpu: str = modal.parameter()
    preload_audio: bool = modal.parameter()

    @modal.enter(snap=True)
    def prepare_dataset(self):
        import zipfile
        import os
        import librosa
        import torch
        import io

        # Load audio files directly from zip into memory as tensors
        print(f"Loading dataset from zip file (preload_audio={self.preload_audio})...")
        zip_path = "/data/processed_data.zip"
        
        if not os.path.exists(zip_path):
            raise RuntimeError(f"Zip file not found at {zip_path}")
        
        if self.preload_audio:
            # Load into memory as tensors
            print("Loading audio files into memory as tensors...")
            self.audio_tensors = []
            self.audio_lengths = []
            self.audio_durations = []
            self.filenames = []
            
            # Read zip file and load audio directly into memory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                audio_files = [f for f in zip_ref.namelist() if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
                print(f"Found {len(audio_files)} audio files in zip")
                
                for i, filename in enumerate(audio_files):
                    if i % 100 == 0:
                        print(f"Loaded {i}/{len(audio_files)} audio files")
                        
                    try:
                        # Read audio file bytes from zip
                        with zip_ref.open(filename) as audio_file:
                            audio_bytes = audio_file.read()
                        
                        # Load audio from bytes using librosa
                        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
                        
                        # Convert to torch tensor
                        audio_tensor = torch.tensor(audio_array, dtype=torch.bfloat16)
                        
                        self.audio_tensors.append(audio_tensor)
                        self.audio_lengths.append(len(audio_array))
                        self.audio_durations.append(len(audio_array) / float(sr))
                        self.filenames.append(os.path.basename(filename))
                        
                    except Exception as e:
                        print(f"Error loading audio file {filename}: {e}")
                        # Add empty tensor as placeholder
                        self.audio_tensors.append(torch.tensor([], dtype=torch.bfloat16))
                        self.audio_lengths.append(0)
                        self.audio_durations.append(0.0)
                        self.filenames.append(os.path.basename(filename))
            
            print(f"Finished loading {len(self.audio_tensors)} audio files into memory")
            
            # Sort all lists by audio duration (longest first)
            print("Sorting audio files by duration (longest first)...")
            combined_data = list(zip(
                self.audio_tensors,
                self.audio_lengths, 
                self.audio_durations,
                self.filenames
            ))
            
            # Sort by duration in descending order (longest first)
            combined_data.sort(key=lambda x: x[2], reverse=True)
            
            # Unpack sorted data back into separate lists
            self.audio_tensors, self.audio_lengths, self.audio_durations, self.filenames = zip(*combined_data)
            
            # Convert back to lists (zip returns tuples)
            self.audio_tensors = list(self.audio_tensors)
            self.audio_lengths = list(self.audio_lengths)
            self.audio_durations = list(self.audio_durations)
            self.filenames = list(self.filenames)
            
            print(f"Sorted by duration: longest={self.audio_durations[0]:.2f}s, shortest={self.audio_durations[-1]:.2f}s")
            
        else:
            # Extract to files and store paths
            print("Extracting audio files and storing file paths...")
            local_data_path = "/tmp/audio_data"
            os.makedirs(local_data_path, exist_ok=True)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_data_path)
            
            # Get file paths and calculate durations for sorting
            audio_file_data = []
            for file in os.listdir(local_data_path):
                if file.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    file_path = os.path.join(local_data_path, file)
                    try:
                        # Load just to get duration for sorting
                        audio_array, sr = librosa.load(file_path, sr=None)
                        duration = len(audio_array) / float(sr)
                        audio_file_data.append((file_path, duration, file))
                    except Exception as e:
                        print(f"Error reading file for sorting {file}: {e}")
                        audio_file_data.append((file_path, 0.0, file))
            
            # Sort by duration (longest first)
            audio_file_data.sort(key=lambda x: x[1], reverse=True)
            
            # Store sorted file paths and metadata
            self.audio_file_paths = [x[0] for x in audio_file_data]
            self.audio_durations = [x[1] for x in audio_file_data]
            self.filenames = [x[2] for x in audio_file_data]
            
            print(f"Extracted and sorted {len(self.audio_file_paths)} audio files")
            if self.audio_durations:
                print(f"Sorted by duration: longest={self.audio_durations[0]:.2f}s, shortest={self.audio_durations[-1]:.2f}s")
        
        # Calculate total audio duration
        total_duration = sum(self.audio_durations)
        print(f"Total audio duration: {total_duration:.2f} seconds")

    @modal.enter(snap=False)
    def prepare_model(self):
        import nemo.collections.asr as nemo_asr
        import torch

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=PARAKEET_MODEL_NAME, map_location=torch.device("cuda:0")
        )
        self.model.to(torch.bfloat16)
        self.model.eval()

        if self.model.cfg.decoding.strategy != "beam":
            self.model.cfg.decoding.strategy = "greedy_batch"
            self.model.change_decoding_strategy(self.model.cfg.decoding)

        with torch.cuda.amp.autocast(enabled=False,dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
            if self.preload_audio:
                warmup_data = self.audio_tensors[:min(BATCH_SIZE*4, len(self.audio_tensors))]
            else:
                warmup_data = self.audio_file_paths[:min(BATCH_SIZE*4, len(self.audio_file_paths))]
            
            self.model.transcribe(
                warmup_data,
                batch_size=BATCH_SIZE,
                num_workers=1,
            )

    @modal.method()
    def run(self, files: List[Path]) -> List[TranscriptionResult]:
        # Batch transcription
        import torch
        import librosa
        import numpy as np
        import os
        
        start_time = time.perf_counter()
        
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
            if self.preload_audio:
                # Use preloaded tensors
                outputs = self.model.transcribe(
                    self.audio_tensors,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                )
            else:
                # Use file paths
                outputs = self.model.transcribe(
                    self.audio_file_paths,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                )
        
        transcription_time = time.perf_counter() - start_time
        
        total_duration = sum(self.audio_durations)
        data_count = len(self.audio_tensors) if self.preload_audio else len(self.audio_file_paths)
        mode_str = "tensors" if self.preload_audio else "file paths"
        
        print(f"Batch transcription time for {data_count} files ({mode_str}) with total duration {total_duration:.2f} seconds:")
        print(f"\t\t{transcription_time:.2f} seconds")
        print("**Real-time factor**:")
        print(f"\t\t{total_duration / transcription_time:.2f}x")

        # Return transcription results
        results = []
        for i, filename in enumerate(self.filenames):
            results.append(TranscriptionResult(
                filename=filename,
                transcription=outputs[i].text,
                transcription_time=transcription_time / len(self.filenames)  # Average time per file
            ))

        return results


def batch_files(files: List[Path], batch_size: int) -> List[List[Path]]:
    """Split files into batches of specified size."""
    batches = []
    for i in range(0, len(files), batch_size):
        batches.append(files[i:i + batch_size])
    return batches
def load_preprocessed_metadata(cache_path: Path) -> Dict[str, BenchmarkResult]:
    """Load preprocessed metadata from JSON file."""
    with open(cache_path, "r") as f:
        data = json.load(f)
    
    # Convert dictionaries back to BenchmarkResult objects
    metadata = {}
    for filename, result_data in data.items():
        metadata[filename] = BenchmarkResult(
            model=result_data["model"],
            filename=result_data["filename"],
            gpu=result_data["gpu"],
            expected_transcription=result_data["expected_transcription"],
            transcription=result_data["transcription"],
            transcription_time=result_data["transcription_time"],
            audio_duration=result_data["audio_duration"],
            wer=result_data["wer"],
        )
    
    print(f"Loaded preprocessed metadata for {len(metadata)} files from {cache_path}")
    return metadata

@app.function(
    volumes={
        DATASET_PATH.as_posix(): dataset_volume,
    },
    image=parakeet_gpu_image,
    timeout=60*60,
)
def benchmark_parakeet(preload_audio: bool = True):

    def preprocess_files(files: List[Path]) -> Dict[str, BenchmarkResult]:
        """Extract metadata and audio duration for all files."""
        import librosa
        import os
        
        # Try to download metadata from volume
        metadata = {}
        try:
            metadata_path = "/data/metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"Loaded metadata for {len(metadata)} files")
            else:
                print("No metadata.json found in volume")
                
        except Exception as e:
            print(f"Could not load metadata: {e}")
            print("Continuing without metadata - will only measure transcription performance")
        
        results = {}
        print(f"Preprocessing {len(files)} files...")
        
        for i, file in enumerate(files):
            if i % 100 == 0:
                print(f"Processed {i}/{len(files)} files")
                
            benchmark_result = BenchmarkResult(
                model=PARAKEET_MODEL_DISPLAY_NAME,
                filename=file.name,
                gpu="",  # Will be set later
            )
            
            # Get expected transcription from metadata if available
            file_metadata = metadata.get(file.name)
            if file_metadata is not None:
                benchmark_result.expected_transcription = file_metadata["transcription"]
                
                # Get audio duration
                try:
                    y, sr = librosa.load(file, sr=None)
                    benchmark_result.audio_duration = len(y) / float(sr)
                except Exception as e:
                    print(f"Error loading audio file {file.name}: {e}")
            else:
                # If no metadata, we can still transcribe but won't have WER
                try:
                    y, sr = librosa.load(file, sr=None)
                    benchmark_result.audio_duration = len(y) / float(sr)
                except Exception as e:
                    print(f"Error loading audio file {file.name}: {e}")
                    
            results[file.name] = benchmark_result
        
        print(f"Finished preprocessing {len(files)} files")
        return results

    def save_preprocessed_metadata(metadata: Dict[str, BenchmarkResult], cache_path: Path):
        """Save preprocessed metadata to JSON file."""
        # Convert BenchmarkResult objects to dictionaries for JSON serialization
        serializable_data = {}
        for filename, result in metadata.items():
            serializable_data[filename] = {
                "model": result.model,
                "filename": result.filename,
                "gpu": result.gpu,
                "expected_transcription": result.expected_transcription,
                "transcription": result.transcription,
                "transcription_time": result.transcription_time,
                "audio_duration": result.audio_duration,
                "wer": result.wer,
            }
        
        with open(cache_path, "w") as f:
            json.dump(serializable_data, f, indent=2)
        print(f"Saved preprocessed metadata for {len(metadata)} files to {cache_path}")

    
        
    files = [
        (DATASET_PATH / Path(f.path)) for f in dataset_volume.listdir("/processed")
    ]

    # Check if preprocessed metadata cache exists
    cache_path = DATASET_PATH / "transcription_metadata.json"
    
    if cache_path.exists():
        print(f"Found cached preprocessed metadata at {cache_path}")
        try:
            file_metadata = load_preprocessed_metadata(cache_path)
            
            # Verify that all current files are in the cache
            cached_filenames = set(file_metadata.keys())
            current_filenames = set(f.name for f in files)
            
            if cached_filenames >= current_filenames:
                print("Cache contains all current files, using cached data")
            else:
                missing_files = current_filenames - cached_filenames
                print(f"Cache missing {len(missing_files)} files: {list(missing_files)[:5]}{'...' if len(missing_files) > 5 else ''}")
                print("Regenerating preprocessed metadata...")
                file_metadata = preprocess_files(files)
                save_preprocessed_metadata(file_metadata, cache_path)
                
        except Exception as e:
            print(f"Error loading cached metadata: {e}")
            print("Regenerating preprocessed metadata...")
            file_metadata = preprocess_files(files)
            save_preprocessed_metadata(file_metadata, cache_path)
    else:
        print("No cached preprocessed metadata found, generating...")
        # Preprocess all files to extract metadata and durations
        file_metadata = preprocess_files(files)
        save_preprocessed_metadata(file_metadata, cache_path)
    
    # Filter files that have valid metadata
    valid_files = [f for f in files if file_metadata[f.name].expected_transcription is not None]
    print(f"Found {len(valid_files)} valid files out of {len(files)} total files")

    GPU_CLASSES = {
        # "cpu": ParakeetCPU,
        **{gpu: ParakeetGPU.with_options(gpu=gpu) for gpu in GPUS},
    }

    for gpu, model_class in GPU_CLASSES.items():
        parakeet = model_class(gpu=gpu, preload_audio=preload_audio)
        
        # Batch the valid files
        file_batches = batch_files(valid_files, BATCH_SIZE*BATCHES_PER_CALL)
        print(f"Processing {len(valid_files)} files in {len(file_batches)} batches of size {BATCH_SIZE}")
        
        # Process batches and collect transcription results
        all_transcription_results = []
        for batch_results in parakeet.run.map(file_batches):
            all_transcription_results.extend(batch_results)
        
        # Combine transcription results with metadata
        import evaluate
        wer_model = evaluate.load("wer")
        
        final_results = []
        for trans_result in all_transcription_results:
            benchmark_result = file_metadata[trans_result.filename]
            benchmark_result.gpu = gpu
            benchmark_result.transcription = trans_result.transcription
            benchmark_result.transcription_time = trans_result.transcription_time
            
            # Calculate WER if we have expected transcription
            if benchmark_result.expected_transcription:
                wer = wer_model.compute(
                    predictions=[trans_result.transcription], 
                    references=[benchmark_result.expected_transcription]
                )
                benchmark_result.wer = wer
            
            final_results.append(benchmark_result)
        
        write_results.remote(final_results, PARAKEET_MODEL_DISPLAY_NAME)

@app.local_entrypoint()
def main():
    import zipfile
    import os
    # zip contents of `tmp_data` dir
    with dataset_volume.batch_upload(force=True) as batch:
        with zipfile.ZipFile("tmp_data.zip", "w") as zipf:
            count = 0
            for file in os.listdir("tmp_data/processed"):
                zipf.write(os.path.join("tmp_data/processed", file), file)
                count += 1

            print(f"Uploaded {count} files to volume")
            batch.put_file("tmp_data.zip", "/processed_data.zip")

    # Example: Run benchmark with preloaded tensors (default)
    print("Running benchmark with preloaded tensors...")
    benchmark_parakeet.remote(preload_audio=True)
    
    # Example: Run benchmark with file paths
    print("Running benchmark with file paths...")
    benchmark_parakeet.remote(preload_audio=False)

if __name__ == "__main__":
    main()