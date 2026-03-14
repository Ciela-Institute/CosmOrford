import argparse
import math
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import datasets
from datasets import Dataset, Features, Array2D, Value, Sequence
from tqdm import tqdm


def _extract_seed(filename: str):
    match = re.search(r"seed=(\d+)", filename)
    return int(match.group(1)) if match else None


def _load_one_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        maps = data["maps"]
        theta = data["theta"]
    return maps, theta


def _normalize_shapes(maps, theta, file_path: Path):
    if maps.ndim == 2:
        maps = maps[np.newaxis, ...]
    if maps.ndim != 3:
        raise ValueError(f"Unexpected maps shape {maps.shape} in {file_path}")

    theta = np.asarray(theta)
    if theta.ndim > 1:
        theta = theta.reshape(-1)
    if theta.shape[0] != 3:
        raise ValueError(f"Unexpected theta shape {theta.shape} in {file_path}")

    if maps.shape[1:] != (1424, 176):
        raise ValueError(f"Unexpected maps spatial shape {maps.shape[1:]} in {file_path}")

    return maps, theta


def _load_and_prepare(fp: Path):
    maps, theta = _load_one_npz(fp)
    maps, theta = _normalize_shapes(maps, theta, fp)

    n_patch = maps.shape[0]
    theta_rep = np.repeat(theta[np.newaxis, :], n_patch, axis=0)

    return maps.astype(np.float16, copy=False), theta_rep.astype(np.float16, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa_dir", type=str, default="scratch/kappa",
                        help="Directory containing kappa .npz files")
    parser.add_argument("--pattern", type=str, default="pmpp_1024_maps_seed=*_kappa.npz",
                        help="Glob pattern for kappa files")
    parser.add_argument("--output_dir", type=str, default="scratch/kappa",
                        help="Directory to save the Hugging Face dataset")
    parser.add_argument("--dataset_name", type=str, default="hf_pmnbody",
                        help="Folder name for the saved dataset")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Number of files per chunk")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of threads for loading files")
    parser.add_argument("--no_sort", action="store_true",
                        help="Disable sorting by seed in filenames")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of files to load")
    parser.add_argument("--start_chunk", type=int, default=0,
                        help="Start processing from this chunk index (0-based)")
    parser.add_argument("--no_chunk_save", action="store_true",
                        help="Disable saving per-chunk datasets to disk")
    parser.add_argument("--chunk_only", action="store_true",
                        help="Only save per-chunk datasets; skip global merge/split to reduce memory")
    parser.add_argument("--from_chunks", action="store_true",
                        help="Build the final dataset by merging existing chunk datasets on disk")
    parser.add_argument("--chunks_dir", type=str, default=None,
                        help="Optional override for chunks directory (defaults to output_dir/dataset_name/chunks)")
    args = parser.parse_args()

    if args.from_chunks:
        chunks_root = Path(args.chunks_dir) if args.chunks_dir else Path(args.output_dir) / args.dataset_name / "chunks"
        if not chunks_root.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_root}")

        chunk_dirs = sorted([p for p in chunks_root.iterdir() if p.is_dir()])
        if not chunk_dirs:
            raise FileNotFoundError(f"No chunk directories found in: {chunks_root}")

        datasets_list = []
        for p in tqdm(chunk_dirs, desc="Loading chunk datasets"):
            datasets_list.append(datasets.load_from_disk(str(p)))

        dataset = datasets.concatenate_datasets(datasets_list)
        dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / args.dataset_name)
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to: {save_path}")
        return

    kappa_dir = Path(args.kappa_dir).expanduser()
    if not kappa_dir.exists():
        raise FileNotFoundError(f"Directory not found: {kappa_dir}")

    files = list(kappa_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {kappa_dir} matching pattern: {args.pattern}")

    if not args.no_sort:
        files = sorted(files, key=lambda p: (_extract_seed(p.name) is None, _extract_seed(p.name), p.name))
    else:
        files = sorted(files)

    if args.limit is not None:
        files = files[:args.limit]

    num_files = len(files)
    num_chunks = math.ceil(num_files / args.chunk_size)

    if args.start_chunk < 0:
        raise ValueError("--start_chunk must be >= 0")
    if args.start_chunk >= num_chunks:
        raise ValueError(f"--start_chunk={args.start_chunk} is >= total chunks ({num_chunks})")

    dataset = None
    chunk_paths = []
    for chunk_idx in tqdm(range(args.start_chunk, num_chunks), desc="Processing chunks"):
        chunk_start = chunk_idx * args.chunk_size
        chunk_end = min((chunk_idx + 1) * args.chunk_size, num_files)
        chunk_files = files[chunk_start:chunk_end]

        chunk_maps = []
        chunk_thetas = []

        if args.num_workers <= 1:
            for fp in tqdm(chunk_files, desc="Loading files", leave=False):
                maps, theta_rep = _load_and_prepare(fp)
                chunk_maps.append(maps)
                chunk_thetas.append(theta_rep)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                for maps, theta_rep in tqdm(
                    pool.map(_load_and_prepare, chunk_files),
                    total=len(chunk_files),
                    desc="Loading files",
                    leave=False,
                ):
                    chunk_maps.append(maps)
                    chunk_thetas.append(theta_rep)

        if not chunk_maps:
            continue

        chunk_maps = np.concatenate(chunk_maps, axis=0)
        chunk_thetas = np.concatenate(chunk_thetas, axis=0).tolist()

        chunk_dataset = Dataset.from_dict(
            {
                "theta": chunk_thetas,
                "maps": chunk_maps,
            },
            features=Features(
                {
                    "theta": Sequence(Value(dtype="float16"), length=3),
                    "maps": Array2D(dtype="float16", shape=(1424, 176)),
                }
            ),
        )

        if not args.no_chunk_save:
            chunks_root = Path(args.output_dir) / args.dataset_name / "chunks"
            chunks_root.mkdir(parents=True, exist_ok=True)
            chunk_path = chunks_root / f"chunk_{chunk_idx:04d}"
            chunk_dataset.save_to_disk(str(chunk_path))
            chunk_paths.append(str(chunk_path))

        if not args.chunk_only:
            if dataset is None:
                dataset = chunk_dataset
            else:
                dataset = datasets.concatenate_datasets([dataset, chunk_dataset])

        # Drop references eagerly to help the GC on large runs
        if args.chunk_only:
            del chunk_maps, chunk_thetas, chunk_dataset

    if args.chunk_only:
        print(f"Chunk datasets saved under: {Path(args.output_dir) / args.dataset_name / 'chunks'}")
        print("Chunk-only mode enabled; skipping global merge/split.")
        return

    if dataset is None:
        raise ValueError("No data loaded; dataset is empty.")

    dataset = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(output_dir / args.dataset_name)
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to: {save_path}")
    if chunk_paths:
        print(f"Chunk datasets saved under: {Path(save_path) / 'chunks'}")


if __name__ == "__main__":
    main()
