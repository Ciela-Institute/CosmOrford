import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset, Features, Sequence, Value, Array2D
import datasets

from tqdm import tqdm


features = Features(
    {
        "kappa": datasets.Array2D(shape=(1424, 176), dtype="float32"),
        "theta": Sequence(Value("float32"), length=3),
    }
)

def sample_generator():

    for i in range(100):
        batch =  np.load(f"GRF/grf_sim_{i}_2.0arcmin_2048nside.npz")
        kappas = batch["train"]
        thetas = batch["label"]
        for sims, theta in zip(kappas, thetas):
            for j in range(len(sims)):

                yield {
                    "kappa": sims[j].astype('float32'), # (1424, 176) float16
                    "theta": theta[1:].astype('float32'), # omega_m, S8, delta_z
                }

cache_dir = "GRF_HF/_cache"
ds = Dataset.from_generator(
    sample_generator,
    features=features,
    cache_dir=str(cache_dir)
)

print(ds)

ds.save_to_disk("GRF_HF")
