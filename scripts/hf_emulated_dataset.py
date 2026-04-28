import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import datasets
from datasets import Dataset, Features, Array2D, Value, Sequence
from cosmoford.emulator.torch_models import build_unet2d_condition_with_y
import torch
import yaml
from cosmoford.emulator.neural_ode import solve_ode_forward
from cosmoford.dataset import reshape_field_numpy, inverse_reshape_field_numpy
from cosmoford.emulator.utils import (
    apply_mask
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
parser.add_argument("--run_dir", type=str, default=None, help="Override: wandb offline run directory")
parser.add_argument("--dataset_name", type=str, default=None, help="Override: output dataset name")
cli = parser.parse_args()

with open(cli.config) as f:
    _cfg = yaml.safe_load(f)
if cli.run_dir is not None:
    _cfg["run_dir"] = cli.run_dir
if cli.dataset_name is not None:
    _cfg["dataset_name"] = cli.dataset_name
args = argparse.Namespace(**_cfg)
print("=== Config ===")
for k, v in _cfg.items():
    print(f"  {k}: {v}")
print("==============")


# log normal dataset
dataset_lognormal = datasets.load_from_disk(args.hf_path)

# Subselect a fraction and choose a specific part of the split
total_rows = len(dataset_lognormal)
fraction = float(args.fraction_of_logn_to_generate)
# Compute part size in rows, at least 1
part_size = max(1, int(np.floor(total_rows * fraction)))
n_parts = int(np.ceil(total_rows / part_size))
part_idx = int(args.part_of_the_batched_dataset_to_generate)
start_row = part_idx * part_size
end_row = min(start_row + part_size, total_rows)

# Apply selection
dataset_lognormal = dataset_lognormal.select(list(range(start_row, end_row)))
dataset_lognormal = dataset_lognormal.with_format('numpy')

print(
    f"Loaded lognormal dataset from {args.hf_path} | total_rows={total_rows} | "
    f"selected_rows=[{start_row}:{end_row}) (size={end_row-start_row}) | "
    f"fraction={fraction}, part={part_idx}/{n_parts-1}"
)

run_dir = Path(args.run_dir)
cfg_path = run_dir / "files" / "config_used.yaml"
ckpt_path = run_dir / "files" / "checkpoints" / args.unet_checkpoint
print(f"Loading model from: {ckpt_path}")


# 1. Reconstruct model from saved config
with open(cfg_path, "r") as f:
    config = yaml.safe_load(f)

# Important: sample_size was forced to [field_npix_x, field_npix_y] in training and y_dim must match.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA if available

model = build_unet2d_condition_with_y(config).to(device)

# 2. Load weights (weights_only=True is optional; requires a recent PyTorch)
try:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
except TypeError: # Older PyTorch versions (e.g., < 1.10) may not support weights_only=True
    print("Warning: PyTorch version might not support 'weights_only=True'. Loading without it.")
    state = torch.load(ckpt_path, map_location=device)

model.load_state_dict(state, strict=True)
model.eval()
print("Model loaded successfully!")

def get_data(batch_logn):
        """Run model forward and return (theta, predicted_maps) with NumPy-first pipeline.

        Expects:
            batch_logn['kappa']: numpy array (B_total, H, W)
            batch_logn['theta']: numpy array (B_total, P)
        Returns numpy arrays with matching first dim B_total.
        """
        theta = np.asarray(batch_logn['theta'])  # (B, P)
        maps = np.asarray(batch_logn['kappa'])   # (B, H, W)
        assert maps.ndim == 3 and theta.ndim == 2, f"Unexpected shapes: maps {maps.shape}, theta {theta.shape}"
        # Reshape to reduced representation in NumPy (B, 1834, 88)
        maps_reduced_np = reshape_field_numpy(maps)
        # Mask inputs in NumPy (trained on masked data)
        maps_mask_np = apply_mask(maps_reduced_np)
        # Call solver: it will convert to torch internally if needed
        x_traj_np = solve_ode_forward(
            maps_mask_np,
            model,
            theta,
            device,
            rtol =args.rtol,
            atol = args.atol,
            nb_steps=args.nb_steps
        )  # (T, B, 1834, 88)
        # Final step, inverse reshape to full size and mask again
        x_final_np = x_traj_np[-1]
        x_full_mask_np = apply_mask(x_final_np)
        x_full_np = inverse_reshape_field_numpy(x_full_mask_np)                 # (B, 1424, 176)
        return theta, x_full_np

print("Data generation...")

# Separate batch size (for simulation) from chunk size (for dataset creation)
# Keep original dataset; each row contains 10 maps.
shape_dataset = dataset_lognormal['kappa'][:10].shape

if len(shape_dataset) == 4:
    len_lognormal_dataset = len(dataset_lognormal*10)
else: 
    len_lognormal_dataset = len(dataset_lognormal)
batch_size = args.batch_size
num_chunks = len_lognormal_dataset // args.chunk_size
num_batches = args.chunk_size // batch_size

nb_of_params_to_infer = 3
field_npix_x = 1424
field_npix_y = 176
# Initialize dataset as None
dataset = None

# Create dataset incrementally by chunks
for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
    chunk_params = []
    chunk_maps = []

    # Generate data for this chunk batch by batch
    for b in tqdm(range(num_batches), desc=f"Generating batches for chunk {chunk_idx + 1}/{num_chunks}"):
        # Select rows first, then slice per-row columns to avoid multi-dim indexing on a column
        if len(shape_dataset) == 3:
            start = chunk_idx * args.chunk_size + b * batch_size
            end = chunk_idx * args.chunk_size + (b + 1) * batch_size
        else:
            start = chunk_idx * args.chunk_size + b * batch_size //10
            end = chunk_idx * args.chunk_size + (b + 1) * batch_size //10
        rows = dataset_lognormal[start:end]
        kapps_list = rows['kappa']
        kapps_arr = np.stack(kapps_list, axis=0)
        thetas = np.asarray(rows['theta'])
        if len(shape_dataset) == 4:
            B_local, N_maps, H_map, W_map = kapps_arr.shape
            kapps_flat = kapps_arr.reshape(B_local * N_maps, H_map, W_map)  # (B*10,H,W)
            thetas_logn_full =thetas[:, 1:]
            thetas_flat = np.repeat(thetas_logn_full, N_maps, axis=0)       # (B*10, P-1)
        else:
            kapps_flat = kapps_arr  # (B,H,W)
            thetas_flat = thetas
        batch_logn = {'kappa': kapps_flat, 'theta': thetas_flat}
        params, kmap = get_data(batch_logn)
        # Convert to numpy arrays
        chunk_params.append(np.array(params))
        chunk_maps.append(np.array(kmap))
    
    if chunk_params:  # Only process if we have data
        # Stack the batches for this chunk and ensure proper numpy arrays
        chunk_params = np.stack(chunk_params, axis=0).reshape([-1, nb_of_params_to_infer])
        chunk_maps = np.stack(chunk_maps, axis=0).reshape([-1, field_npix_x, field_npix_y])
        
        chunk_maps = np.array(chunk_maps, dtype=np.float16)
        chunk_params = np.array(chunk_params, dtype=np.float16).tolist()

        # Create chunk dataset
        chunk_dataset = Dataset.from_dict({
            'theta': chunk_params,
            'maps': chunk_maps,
        }, features = Features({
            'theta': Sequence(Value(dtype='float16'), length=nb_of_params_to_infer),
            'maps': Array2D(dtype='float16', shape=(field_npix_x, field_npix_y)),
        }))

        # For the first chunk, create the initial dataset
        if dataset is None:
            dataset = chunk_dataset
        else:
            # Concatenate with existing dataset
            dataset = datasets.concatenate_datasets([dataset, chunk_dataset])


dataset = dataset.shuffle(seed=98)

# Save the dataset
save_path = str(Path(args.output_dir) / args.dataset_name)
dataset.save_to_disk(save_path)
print(f"Dataset saved to: {save_path}")
print("Done!")
