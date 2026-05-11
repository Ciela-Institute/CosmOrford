"""Merge per-budget emulated datasets into a single DatasetDict.

Each split corresponds to a simulation budget, e.g. "budget_100".
Individual datasets are produced by hf_emulated_dataset.py.

Usage:
    python scripts/merge_emulated_budget_datasets.py --config configs/merge_emulated_budget_datasets.yaml
"""
import argparse
import yaml
from pathlib import Path
from datasets import load_from_disk, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
cli = parser.parse_args()

with open(cli.config) as f:
    cfg = yaml.safe_load(f)

input_dir = Path(cfg["input_dir"])
output_path = Path(cfg["output_dir"]) / cfg["output_dataset_name"]
budgets = cfg["budgets"]
dataset_name = cfg["dataset_name"]

splits = {}
for budget in budgets:
    path = input_dir / dataset_name.format(budget=budget)
    print(f"Loading budget={budget} from {path} ...")
    splits[f"budget_{budget}"] = load_from_disk(str(path))

dataset_dict = DatasetDict(splits)
print(f"\nSaving merged DatasetDict to {output_path} ...")
dataset_dict.save_to_disk(str(output_path))
print("Done!")
print(dataset_dict)
