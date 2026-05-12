from cosmoford import SURVEY_MASK, THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from tqdm import tqdm
import yaml

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

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Loading the dataset and selecting N=num_obs observations
    num_obs = args.num_obs 
    holdout_path = args.holdout_path
    holdout_dset = load_from_disk(holdout_path)
    dset = holdout_dset["train"].select(range(num_obs))
    dset = dset.with_format("numpy")
    
    kappa = np.array(dset["kappa"])
    theta = np.array(dset["theta"])

    COMPRESSOR_CKPT_PATH = Path(args.compressor_ckpt_path)
    best_compressor_ckpt = find_best_checkpoint(budget = BUDGET, checkpoints_path = COMPRESSOR_CKPT_PATH)
    compressor =  RegressionModelNoPatch.load_from_checkpoint(best_compressor_ckpt, map_location=device)

    # getting the summary dim from the config.yaml in the compressor folder directory. 
    compressor_cfg = load_yaml(COMPRESSOR_CKPT_PATH / f"budget-{BUDGET}" / "config.yaml")
    context_dim = compressor_cfg["model"]["init_args"]["summary_dim"]

    # Loading flow    
    FLOW_CKPT_PATH = f"{args.flow_ckpt_path}/budget-{BUDGET}/npe_flow.pt"    
    flow_weights = torch.load(FLOW_CKPT_PATH)
    flow = build_flow(param_dim=2, context_dim=context_dim).to(device)
    flow.load_state_dict(flow_weights)

    
    # Running the posterior sampling
    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    
    num_samples = args.num_samples
    normalize = False
    posterior_samples = []
    with torch.no_grad():
        for kappa_i in tqdm(kappa):
            kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]
            noisy = kappa_reshaped # kappa maps from the holdout dataset are already noisy and masked
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            s = compressor.compress(x)  # (1, 8)
    
            # Normalize before feeding to flow
            if normalize:
                s = (s - s_mean.to(device)) / s_std.to(device)
    
            # Sample posterior
            samples = flow.sample(num_samples, context=s)  # (1, N, 2)
            samples = samples.squeeze(0).cpu().numpy()  # (N, 2)
    
            # Unnormalize to physical parameters
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]
            posterior_samples.append(samples_phys)

    
    posterior_samples = np.array(posterior_samples) # (N_obs, N_posterior_samples, 2)

    #### COMPUTE MIRA SCORE ####
    
    #### SAVE MIRA-SCORE + POSTERIOR_SAMPLES + THETA_TRUE + CKPT DIRS FOR COMPRESSOR AND FLOW ####
    results_dir = args.results_dir
    mira = None


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
    


if __name__ == "main": 

    import argparse
    
    parser = argparse.ArgumentParser(
        description="MIRA Score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compressor_ckpt_path", required=True,
                        help="Path where chec")
    parser.add_argument("--flow_ckpt_path", required=True,
                        help="Path where chec")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to holdout DatasetDict (save_to_disk format, 'train' and 'fiducial' splits)")
    args = parser.parse_args()