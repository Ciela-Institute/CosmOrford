from cosmoford import SURVEY_MASK, THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from tqdm import tqdm
import torch 
import numpy as np 
import yaml
from mira_score import get_device, mira, mira_bootstrap

def load_yaml(path): 
    with open(path) as file: 
        dict_yaml = yaml.safe_load(file)
    return dict_yaml

def save_yaml(path, data):
    with open(path, 'w') as file:
        yaml.dump(data, file)

"""
Example python script usage: 
Doing: 

python mira_score_npe.py --budget=1000 --compressor_ckpt_path=.../budget_scan --flow_ckpt_path=.../npe_results --holdout_path=... --normalize=False --num_samples=10000 --num_obs=200

will: 
1) Compress `num_obs` kappa maps from the holdout dataset loaded from the directory `holdout_path` using the compressor trained with `budget` training samples 
2) Sample a NPE given the summaries to create an array of posterior samples of shape `(num_obs, num_samples, 2)`. Also store the ground-truth parameters in an array `theta` 
of shape `(num_obs, 2)`
3) Compute MIRA score 
4) Save posterior samples (maybe we should plot them for different observations?), theta, MIRA score 
"""

from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from datasets import load_from_disk
from glob import glob

device = get_device()

def find_best_checkpoint(budget: int, checkpoints_path: Path, offline: bool = False) -> str:
    """Find the best compressor checkpoint for a given budget.

    Strategy:
    1. Parse val_mse from Lightning checkpoint filenames, pick lowest
    2. Fall back to last.ckpt
    3. Fall back to W&B artifact download (skipped when offline=True)
    """
    import re

    checkpoint_dir = checkpoints_path / f"budget-{budget}"
    checkpoints = str(checkpoint_dir)
    # print(checkpoints)
    checkpoints = glob(checkpoints + "/**/*.ckpt", recursive=True)

    # for ckpt in checkpoint_dir.glob("*.ckpt"): 
    #     print(ckpt)
    # Strategy 1: Parse val_mse from checkpoint filenames
    if checkpoint_dir.exists():
        best_path = None
        best_logp = float("inf")

        for ckpt in checkpoints:
            print(ckpt)
            if ckpt == "last.ckpt":
                continue
            # match = re.search(r"val_log_prob=([\d.]+)", ckpt)
            match = re.search(r"val_log_prob=([-+]?\d*\.?\d+)", ckpt)
            if match:
                logp = float(match.group(1))
                if logp < best_logp:
                    best_logp = logp
                    best_path = str(ckpt)

        if best_path is not None:
            print(f"Found best checkpoint for budget-{budget}: {best_path} (val_log_prob={best_logp:.6f})")
            return best_path

        # Strategy 2: Fall back to last.ckpt
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            print(f"Using last.ckpt for budget-{budget}")
            return str(last_ckpt)

    if offline:
        raise FileNotFoundError(
            f"No local checkpoint found for budget-{budget} in {checkpoint_dir} "
            f"and --offline is set (W&B download disabled)."
        )

    # Strategy 3: W&B fallback
    print(f"No local checkpoint for budget-{budget}, trying W&B...")
    import wandb

    api = wandb.Api()
    runs = api.runs(
        "cosmostat/neurips-wl-challenge",
        filters={"tags": "budget-scan", "display_name": f"budget-{budget}"},
    )
    for run in runs:
        for art in run.logged_artifacts():
            if art.type == "model":
                art_dir = art.download(root=str(checkpoint_dir))
                for f in Path(art_dir).glob("**/*.ckpt"):
                    print(f"Downloaded W&B artifact for budget-{budget}: {f}")
                    return str(f)

    raise FileNotFoundError(f"No checkpoint found for budget-{budget}")

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Loading the dataset and selecting N=num_obs observations
    print('Loading holdout dataset...')
    num_obs = args.num_obs 
    holdout_path = args.holdout_path
    holdout_dset = load_from_disk(holdout_path)
    dset = holdout_dset["train"].select(range(num_obs))
    dset = dset.with_format("numpy")
    
    kappa = np.array(dset["kappa"])
    theta = np.array(dset["theta"])

    print(f"Selected {len(dset)} kappa maps from the holdout.")

    print("Loading and Initializing Compressor + Flow...")
    COMPRESSOR_CKPT_PATH = Path(args.compressor_ckpt_path)
    budget = args.budget
    best_compressor_ckpt = find_best_checkpoint(budget = budget, checkpoints_path = COMPRESSOR_CKPT_PATH)
    compressor =  RegressionModelNoPatch.load_from_checkpoint(best_compressor_ckpt, map_location=device)

    # getting the summary dim from the config.yaml in the compressor folder directory. 
    compressor_cfg = load_yaml(COMPRESSOR_CKPT_PATH / f"budget-{budget}" / "config.yaml")
    context_dim = compressor_cfg["model"]["init_args"]["summary_dim"]

    # Loading flow    
    FLOW_CKPT_PATH = f"{args.flow_ckpt_path}/budget-{budget}/npe_flow.pt"    
    flow_weights = torch.load(FLOW_CKPT_PATH)
    flow = build_flow(param_dim=2, context_dim=context_dim).to(device)
    flow.load_state_dict(flow_weights)
    print("Nets loaded !")

   
    # Running the posterior sampling
    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    
    num_samples = args.num_samples
    normalize = False
    posterior_samples = []

    print(f"Sampling the NPE; Creating {num_samples} posterior samples per observation for {num_obs}")
    with torch.no_grad():
        for kappa_i in tqdm(kappa):
            kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]
            noisy = kappa_reshaped # kappa maps from the holdout dataset are already noisy and masked
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            s = compressor.compress(x)  # (1, 8)
    
            # Normalize before feeding to flow
            if normalize:
                s = (s - s.mean().to(device)) / s.std().to(device)
    
            # Sample posterior
            samples = flow.sample(num_samples, context=s)  # (1, N, 2)
            samples = samples.squeeze(0).cpu().numpy()  # (N, 2)
    
            # Unnormalize to physical parameters
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]
            posterior_samples.append(samples_phys)

    
    posterior_samples = np.array(posterior_samples) # (N_obs, N_posterior_samples, 2)
    # Make posterior_samples (1, N_obs, N_posterior_samples, 2)
    posterior_samples = np.expand_dims(posterior_samples, axis=0) # (1, N_obs, N_posterior_samples, 2)

    #### COMPUTE MIRA SCORE ####
    mira_score_mean, mira_score_std = mira(
        torch.tensor(theta[:, :2]).to(device), 
        torch.tensor(posterior_samples).to(device), 
        num_runs=100, 
        norm=True
    )
    
    #### SAVE MIRA-SCORE + POSTERIOR_SAMPLES + THETA_TRUE + CKPT DIRS FOR COMPRESSOR AND FLOW ####
    print("Saving results...")
    results_dir = args.results_dir
    mira_score = {
    "mean": mira_score_mean.detach().cpu().numpy() if torch.is_tensor(mira_score_mean) else mira_score_mean,
    "std": mira_score_std.detach().cpu().numpy() if torch.is_tensor(mira_score_std) else mira_score_std,
}


    import os
    os.makedirs(args.results_dir,exist_ok=True)
    np.savez(
        results_dir + "/mira_data.npz",
        posterior_samples = posterior_samples, # (N_obs, N_posterior_samples, 2)
        theta = theta, 
        mira_score = mira_score
    )

    net_info = {
        "compressor_cfg": compressor_cfg, 
        "flow_cfg": {
            "param_dim": 2, 
            "context_dim": context_dim
        }
    }
    save_yaml(results_dir + "/net_cfg.yaml", data = net_info)

    # Save Specific MIRA Config ?
    
    print(f"Results saved at {results_dir}")

if __name__ == "__main__": 

    import argparse
    
    parser = argparse.ArgumentParser(
        description="MIRA Score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--budget", required = True, type = int, help = "Training Budget used for the Compressor and the NPE training.")
    parser.add_argument("--compressor_ckpt_path", required=True,
                        help="Checkpoint path for the compressor")
    parser.add_argument("--flow_ckpt_path", required=True,
                        help="Checkpoints path for the npe model")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to holdout DatasetDict (save_to_disk format, 'train' and 'fiducial' splits)")
    parser.add_argument("--normalize", required=False, default = False, type = bool,
                        help="Whether to normalize the summaries before feeding them to the normalizing flow. Defaults to False.")
    parser.add_argument("--num_samples", required=False, default = 10_000, type = int,
                        help="Number of posterior samples to create per observation. Defaults to 10 000.")
    parser.add_argument("--num_obs", required=False, default = 200, type = int,
                        help="Number of observations to take from the holdout dataset. Defaults to 200.")
    parser.add_argument("--results_dir", required=True,
                        help="Checkpoint path for the compressor")
    parser.add_argument("--seed", required=False, default = 42, type = int,
                        help="Random seed to run the code")

    args = parser.parse_args()
    main(args)