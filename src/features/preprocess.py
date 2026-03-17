import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.features.extractor import AudioFeatureExtractor


def get_audio_path(data_dir: str, track_id: int) -> str:
    """FMA folder structure: data_dir/NNN/NNNNNN.mp3"""
    tid_str = f"{track_id:06d}"
    folder = tid_str[:3]
    return os.path.join(data_dir, folder, f"{tid_str}.mp3")


def process_track(args):
    """Worker function for multiprocessing."""
    track_id, audio_path, output_dir, config = args
    extractor = AudioFeatureExtractor(**config)
    out_path = os.path.join(output_dir, f"{track_id:06d}.npy")

    if os.path.exists(out_path):
        return track_id, True

    try:
        feat = extractor.extract(audio_path, feature_type="mel_spectrogram")
        np.save(out_path, feat)
        return track_id, True
    except Exception:
        return track_id, False


def preprocess_dataset(
    track_ids: list,
    data_dir: str,
    output_dir: str,
    extractor_config: dict,
    num_workers: int = 4,
) -> dict:
    """Batch-process audio files into feature arrays."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tasks = []
    for tid in track_ids:
        audio_path = get_audio_path(data_dir, tid)
        if os.path.exists(audio_path):
            tasks.append((tid, audio_path, output_dir, extractor_config))

    results = {"success": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_track, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            _, ok = future.result()
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1

    return results
