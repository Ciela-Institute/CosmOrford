#!/usr/bin/env python3
import argparse
import math
import sys


def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_maps(torch_mod, batch: int, ny: int, nx: int, device):
    y = torch_mod.linspace(-1.0, 1.0, ny, device=device)
    x = torch_mod.linspace(-1.0, 1.0, nx, device=device)
    yy, xx = torch_mod.meshgrid(y, x, indexing="ij")
    base = (
        0.6 * torch_mod.exp(-8.0 * (xx**2 + yy**2))
        + 0.3 * torch_mod.sin(6.0 * math.pi * xx)
        + 0.2 * torch_mod.cos(4.0 * math.pi * yy)
    )
    noise = 0.1 * torch_mod.randn(batch, ny, nx, device=device)
    return base.unsqueeze(0).repeat(batch, 1, 1) + noise


def reshape_field_local(kappa):
    """Reduce full map (B, 1424, 176) to challenge layout (B, 1834, 88)."""
    import torch
    return torch.cat([kappa[:, :, :88], kappa[:, 620:1030, 88:]], dim=1)


def cosine_mean(torch_mod, a, b):
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1) + 1e-8
    return (num / den).mean().item()


def relative_l2_mean(torch_mod, ref, other):
    rel = (other - ref).norm(dim=1) / (ref.norm(dim=1) + 1e-8)
    return rel.mean().item()


def main():
    try:
        import torch
        import numpy as np
        from cosmoford import SURVEY_MASK
        from cosmoford.summaries import (
            compute_scattering_batch,
            scattering_n_coefficients,
            scattering_order_slices,
        )
    except ModuleNotFoundError as exc:
        print(
            f"ERROR: missing dependency '{exc.name}'. Activate the training environment "
            "before running WST diagnostics.",
            file=sys.stderr,
        )
        raise

    parser = argparse.ArgumentParser(description="WST diagnostic sweep")
    parser.add_argument("--J", default="3,4,5", help="Comma-separated scattering J values")
    parser.add_argument("--L", default="4,6,8,12,16", help="Comma-separated scattering L values")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ny", type=int, default=1834)
    parser.add_argument("--nx", type=int, default=88)
    parser.add_argument("--full-ny", type=int, default=1424)
    parser.add_argument("--full-nx", type=int, default=176)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--normalization", default="log1p_zscore", choices=["log1p_zscore", "zscore", "none"])
    parser.add_argument("--feature-pooling", default="mean_std", choices=["mean", "mean_std"])
    parser.add_argument("--mask-pooling", default="soft", choices=["soft", "hard"])
    parser.add_argument("--compare-reduced-full", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    maps = build_maps(torch, args.batch_size, args.ny, args.nx, device)
    js = parse_int_list(args.J)
    ls = parse_int_list(args.L)

    reduced_mask_np = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])
    reduced_mask = torch.tensor(reduced_mask_np, device=device, dtype=maps.dtype)
    full_mask = torch.tensor(SURVEY_MASK, device=device, dtype=maps.dtype)

    print(
        "J,L,coverage_frac,coeff_expected,coeff_actual,"
        "order0_mean,order1_mean,order2_mean,"
        "nan_count,mean_raw,std_raw,mean_norm,std_norm,"
        "geom_cos,geom_rel_l2,shift_delta_red,shift_delta_full"
    )
    for j in js:
        if 2 ** j > min(args.ny, args.nx):
            print(f"{j},NA,0,0,0,NA,NA,NA,0,NA,NA,NA,NA,NA,NA,NA")
            continue
        for l in ls:
            raw = compute_scattering_batch(
                maps,
                J=j,
                L=l,
                normalize=False,
                normalization="none",
                feature_pooling=args.feature_pooling,
            )
            normed = compute_scattering_batch(
                maps,
                J=j,
                L=l,
                normalize=True,
                normalization=args.normalization,
                feature_pooling=args.feature_pooling,
            )
            expected = scattering_n_coefficients(j, l, feature_pooling=args.feature_pooling)
            slices = scattering_order_slices(j, l)
            o0 = raw[:, slices["order0"]].mean().item()
            o1 = raw[:, slices["order1"]].mean().item()
            o2 = raw[:, slices["order2"]].mean().item()
            coverage_frac = (args.nx // (2 ** j)) * (2 ** j) / args.nx

            geom_cos = float("nan")
            geom_rel = float("nan")
            shift_red = float("nan")
            shift_full = float("nan")
            if args.compare_reduced_full and 2 ** j <= min(args.full_ny, args.full_nx):
                full_maps = build_maps(torch, args.batch_size, args.full_ny, args.full_nx, device)
                red_maps = reshape_field_local(full_maps)

                full_maps = full_maps * full_mask.unsqueeze(0)
                red_maps = red_maps * reduced_mask.unsqueeze(0)

                full_feat = compute_scattering_batch(
                    full_maps,
                    J=j,
                    L=l,
                    normalize=True,
                    normalization=args.normalization,
                    mask=full_mask,
                    mask_pooling=args.mask_pooling,
                    feature_pooling=args.feature_pooling,
                )
                red_feat = compute_scattering_batch(
                    red_maps,
                    J=j,
                    L=l,
                    normalize=True,
                    normalization=args.normalization,
                    mask=reduced_mask,
                    mask_pooling=args.mask_pooling,
                    feature_pooling=args.feature_pooling,
                )
                geom_cos = cosine_mean(torch, full_feat, red_feat)
                geom_rel = relative_l2_mean(torch, full_feat, red_feat)

                red_shift = torch.roll(red_maps, shifts=1, dims=1)
                full_shift = torch.roll(full_maps, shifts=1, dims=1)
                red_shift_feat = compute_scattering_batch(
                    red_shift,
                    J=j,
                    L=l,
                    normalize=True,
                    normalization=args.normalization,
                    mask=reduced_mask,
                    mask_pooling=args.mask_pooling,
                    feature_pooling=args.feature_pooling,
                )
                full_shift_feat = compute_scattering_batch(
                    full_shift,
                    J=j,
                    L=l,
                    normalize=True,
                    normalization=args.normalization,
                    mask=full_mask,
                    mask_pooling=args.mask_pooling,
                    feature_pooling=args.feature_pooling,
                )
                shift_red = relative_l2_mean(torch, red_feat, red_shift_feat)
                shift_full = relative_l2_mean(torch, full_feat, full_shift_feat)

            print(
                f"{j},{l},{coverage_frac:.4f},{expected},{raw.shape[1]},"
                f"{o0:.6f},{o1:.6f},{o2:.6f},"
                f"{torch.isnan(raw).sum().item()},"
                f"{raw.mean().item():.6f},{raw.std().item():.6f},"
                f"{normed.mean().item():.6f},{normed.std().item():.6f},"
                f"{geom_cos:.6f},{geom_rel:.6f},{shift_red:.6f},{shift_full:.6f}"
            )


if __name__ == "__main__":
    main()
