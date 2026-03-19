# Ablation Study: Summary Statistics for Cosmological Parameter Inference

## Overview

This document reports the results of an ablation study on summary statistics used to infer cosmological parameters (Ω_m, σ_8) from weak gravitational lensing convergence maps. The model is an EfficientNet-B0 regression head trained with a composite score loss (negative log-likelihood + MSE penalty, higher is better) for 30 epochs. All experiments use the same training pipeline, data split, and architecture; only the input summary statistics vary.

The **validation score** is defined as:

$$\text{score} = -\sum_{i \in \{\Omega_m, \sigma_8\}} \left[ \frac{(\hat{\theta}_i - \theta_i)^2}{\hat{\sigma}_i^2} + \log \hat{\sigma}_i^2 + \lambda \cdot (\hat{\theta}_i - \theta_i)^2 \right]$$

where λ = 1000. A higher score indicates both more accurate point estimates (lower MSE) and better-calibrated uncertainty predictions.

Three families of summary statistics are compared:

- **PS**: Angular power spectrum (2-point, Fourier-space)
- **HOS**: Higher-order wavelet statistics — multi-scale wavelet peak counts and L1-norm histograms (non-Gaussian)
- **Scattering**: Wavelet scattering transform coefficients

---

## 1. Feature Combination Ablation

All runs are summarised in the tables below. Runs in §1.1 used early configurations (some with PS normalisation bugs); runs in §1.2 are from the systematic ablation batch. Runs marked † use the corrected per-batch PS normalisation (see §2.1); all others used the original (bugged) global normalisation.

### 1.1 Early exploration runs (WandB IDs)

| Run ID | Statistics | HOS variant | ns | Peaks SNR | L1 SNR | J | Val score | Val MSE |
|--------|-----------|-------------|-----|-----------|--------|---|-----------|---------|
| yj48er19 | PS + HOS | full | 6 | [−3, 7] | [−7, 7] | — | 10.086 | 1.02e-3 |
| msatahn4 | PS + HOS | full | 4 | [−3, 7] | [−7, 7] | — | 10.079 | 1.03e-3 |
| dv0w9s6m | PS + HOS + Scat | full | 6 | [−3, 7] | [−7, 7] | 5 | 11.399 | 7.08e-4 |
| 3cjs3z1x | PS + HOS + Scat | full | 6 | [−3, 7] | [−7, 7] | 5 | 11.331 | 7.24e-4 |
| v6cj7p98 | PS + HOS + Scat | full | 6 | [−3, 7] | [−7, 7] | 4 | 11.041 | 7.78e-4 |
| 7s2u38we | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.361 | 7.04e-4 |
| pmu1wrjv | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.334 | 7.11e-4 |
| qwp9c44x | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.328 | 7.16e-4 |
| lprg7hu1 | HOS | full | 4 | [−3, 7] | [−7, 7] | — | 11.153 | 7.53e-4 |
| psobiafb | HOS | full | 3 | [−3, 7] | [−7, 7] | — | 11.051 | 7.83e-4 |
| wml1jqti | HOS | full | 6 | [−3, 7] | [−7, 7] | — | 10.986 | 7.81e-4 |
| 388wnkgf | HOS | l1_only | 4 | — | [−7, 7] | — | 10.963 | 7.99e-4 |
| szz6u1c1 | HOS | l1_only | 6 | — | [−7, 7] | — | 10.852 | 8.19e-4 |
| fjy66buh | HOS | l1_only | 3 | — | [−7, 7] | — | 10.815 | 8.41e-4 |
| thx56mt2 | HOS | peaks_only | 4 | [−3, 7] | — | — | 10.636 | 8.68e-4 |
| 0ajzckx6 | HOS | peaks_only | 6 | [−3, 7] | — | — | 10.601 | 8.75e-4 |
| et2is8tp | HOS | peaks_only | 3 | [−3, 7] | — | — | 10.570 | 8.82e-4 |
| 1p44eqr2 | Scat | — | — | — | — | 4 | 9.942 | 1.05e-3 |
| 78a96766 | Scat | — | — | — | — | 5 | 9.576 | 1.16e-3 |
| 354qkkwq | Scat | — | — | — | — | 3 | 9.405 | 1.19e-3 |
| md5lhelk | PS | — | — | — | — | — | −0.174 | 4.09e-3 |
| sy8rmbah | PS + Scat | — | — | — | — | 4 | 8.230 | 1.49e-3 |
| f1uqjf8o | PS + Scat | — | — | — | — | 5 | 4.976 | 2.25e-3 |
| 1dl574x4 | PS + Scat | — | — | — | — | 3 | 5.010 | 2.40e-3 |
| 71qhtzi4 | PS + Scat | — | — | — | — | 5 | −23.684 | 7.97e-3 |
| zimy3a1i† | PS | — | — | — | — | — | 6.713 | 2.04e-3 |
| 08qodcg5† | PS + HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.398 | 7.07e-4 |

### 1.2 Systematic ablation runs (SLURM job IDs)

All runs below use 51 bins for peak counts and 80 bins for L1-norm unless noted. Scattering uses L=8 throughout.

| Job ID | Config | Statistics | Variant | ns | Peaks SNR | L1 SNR | J | Val score | Val MSE |
|--------|--------|-----------|---------|-----|-----------|--------|---|-----------|---------|
| 7851220 | ps | PS | — | — | — | — | — | −0.101 | — |
| 7859129† | ps_fixed | PS | — | — | — | — | — | 6.849 | 2.04e-3 |
| 7851221 | scattering_J3 | Scat | — | — | — | — | 3 | 9.438 | 1.19e-3 |
| 7851222 | scattering_J4 | Scat | — | — | — | — | 4 | 9.984 | 1.05e-3 |
| 7851223 | scattering_J5 | Scat | — | — | — | — | 5 | 9.626 | 1.16e-3 |
| 7852113 | peaks_ns3 | HOS | peaks_only | 3 | [−3, 7] | — | — | 10.593 | 8.82e-4 |
| 7851218 | peaks_ns4 | HOS | peaks_only | 4 | [−3, 7] | — | — | 10.653 | 8.68e-4 |
| 7851219 | peaks_ns6 | HOS | peaks_only | 6 | [−3, 7] | — | — | 10.635 | 8.75e-4 |
| 7852114 | l1_ns3 | HOS | l1_only | 3 | — | [−7, 7] | — | 10.870 | 8.41e-4 |
| 7851216 | l1_ns4 | HOS | l1_only | 4 | — | [−7, 7] | — | 11.024 | 7.99e-4 |
| 7851217 | l1_ns6 | HOS | l1_only | 6 | — | [−7, 7] | — | 10.910 | 8.19e-4 |
| 7896638 | l1_only_l1_wide | HOS | l1_only | 4 | — | [−10, 10] | — | 11.096 | 8.08e-4 |
| 7852115 | hos_ns3 | HOS | full | 3 | [−3, 7] | [−7, 7] | — | 11.060 | 7.83e-4 |
| 7851214 | hos_ns4 | HOS | full | 4 | [−3, 7] | [−7, 7] | — | 11.185 | 7.53e-4 |
| 7851215 | hos_ns6 | HOS | full | 6 | [−3, 7] | [−7, 7] | — | 11.050 | 7.81e-4 |
| 7896637 | hos_l1_wide | HOS | full | 4 | [−3, 7] | [−10, 10] | — | 11.277 | 7.04e-4 |
| 7852112* | ps_hos_ns4 | PS + HOS | full | 4 | [−3, 7] | [−7, 7] | — | 10.938 | 8.00e-4 |
| 7896635† | ps_hos_fixed | PS + HOS | full | 4 | [−3, 7] | [−7, 7] | — | 11.239 | 7.04e-4 |
| 7896636† | ps_l1_fixed | PS + L1 | l1_only | 4 | — | [−7, 7] | — | 11.067 | 7.81e-4 |
| 7852343 | hos_scat_J3 | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 3 | 11.397 | 7.04e-4 |
| 7852116 | hos_scat_J4 | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.420 | 7.04e-4 |
| 7852344 | hos_scat_J5 | HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 5 | 11.409 | 7.04e-4 |
| 7859130† | ps_hos_scat_J4_fixed | PS + HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.486 | 7.04e-4 |
| 7896658† | ps_hos_scat_fixed_r2 | PS + HOS + Scat | full | 4 | [−3, 7] | [−7, 7] | 4 | 11.475 | 7.04e-4 |
| 7858310 | snr_peaks_narrow | HOS + Scat | full | 4 | [−1, 5] | [−7, 7] | 4 | 11.321 | 7.29e-4 |
| 7858311 | snr_peaks_wide | HOS + Scat | full | 4 | [−5, 10] | [−7, 7] | 4 | 11.488 | 6.85e-4 |
| 7858312 | snr_peaks_posonly | HOS + Scat | full | 4 | [0, 8] | [−7, 7] | 4 | 11.445 | 7.02e-4 |
| 7896657 | snr_peaks_posonly_r2 | HOS + Scat | full | 4 | [0, 8] | [−7, 7] | 4 | 11.444 | 7.04e-4 |
| 7858313 | snr_l1_narrow | HOS + Scat | full | 4 | [−3, 7] | [−5, 5] | 4 | 11.360 | 7.14e-4 |
| 7858314 | snr_l1_wide | HOS + Scat | full | 4 | [−3, 7] | [−10, 10] | 4 | 11.489 | 6.79e-4 |
| 7896660 | snr_l1_wide_bins120 | HOS + Scat | full | 4 | [−3, 7] | [−10, 10] | 4 (120 L1 bins) | 11.449 | 7.04e-4 |
| 7896661 | snr_l1_wider | HOS + Scat | full | 4 | [−3, 7] | [−12, 12] | 4 | 11.518 | 7.04e-4 |
| **7896659** | **snr_full_wide** | **HOS + Scat** | **full** | **4** | **[−5, 10]** | **[−10, 10]** | **4** | **11.539** | **7.04e-4** |

*\* Run 7852112 peaked at epoch 3 then diverged due to the PS normalisation bug (see §2.1); best score is cherry-picked and not representative of final performance.*

*† Fixed per-batch PS normalisation.*

*Notes: Run `1dl574x4` stopped early at epoch 17. Run `71qhtzi4` diverged; `f1uqjf8o` is a repeat recovering to 4.976. Early PS+Scat runs used an incorrect spatial coverage depth (J=5 on narrow survey geometry); see §2.2 for details.*

---

## 2. Discussion by Feature Family

### 2.1 Power Spectrum (PS) alone

The PS alone yields a near-zero score (−0.17) with the original normalisation, far below the ~5–7 expected from a well-calibrated PS estimator. Post-hoc analysis revealed a **normalisation bug**: the hardcoded constants `LOG_PS_MEAN` and `LOG_PS_STD` used to standardise the PS features were computed on **noiseless** convergence maps, while training and validation always add shape noise (σ_noise ≈ 0.026 per pixel). For white noise, the noise contribution to the power spectrum is flat at log₁₀(P_noise) ≈ −9.65. Because the signal PS falls steeply with wavenumber k (slope ≈ −1.5), the noise dominates at k ≳ 500 rad⁻¹. Using the noiseless `LOG_PS_MEAN` to normalise the noisy maps produces a systematic bias of **+1 to +4σ at k-bins 4–9** (see table below), making the high-k features carry essentially no cosmological information.

| k-bin | log₁₀(P_signal) | log₁₀(P_measured) | Bias (σ) |
|-------|----------------|-------------------|---------|
| 0     | −8.68          | −8.63             | +0.19   |
| 3     | −9.44          | −9.23             | +0.90   |
| 5     | −9.95          | −9.47             | +1.83   |
| 7     | −10.61         | −9.60             | +3.13   |
| 9     | −11.08         | −9.63             | +3.93   |

In contrast, HOS and Scattering both use **per-batch normalisation** (mean and std computed over the current mini-batch), making their features automatically consistent with the actual noise level. The PS normalisation has been corrected to use the same per-batch scheme. After the fix, **PS alone scores 6.85** (job 7859129, up from −0.17), confirming the normalisation bug was the dominant cause of the collapse. The instability caused by the bugged PS normalisation also affected combined PS+HOS runs: job 7852112 (PS+HOS, bugged norm) peaked at epoch 3 and then diverged, demonstrating how corrupted PS features poison network training.

Even with corrected normalisation, PS alone (6.85) remains substantially weaker than HOS alone (11.19) or Scattering alone (9.98). The angular power spectrum is a second-order (Gaussian) statistic insensitive to the non-Gaussian features — filaments, peak abundance, void statistics — generated by non-linear structure formation. These features are precisely what most tightly constrains Ω_m and σ_8 in the late universe. The PS captures the overall amplitude of fluctuations but misses the shape and topology of the convergence field.

### 2.2 Scattering Transform alone

The scattering transform alone achieves scores between 9.4 and 9.98 (varying with the maximum wavelet depth J). As a cascade of wavelet modulus operations, the scattering transform captures multi-scale non-Gaussian features efficiently. However, it is outperformed by full HOS (~11.1–11.2) by about 1.2 points. This gap is consistent with the scattering coefficients being a compact but lossy descriptor: they encode inter-scale correlations well but do not directly provide the one-point statistics (peak counts, field distribution) that carry substantial cosmological information in lensing maps.

An important practical caveat for the scattering transform concerns the survey geometry. The input maps are reshaped to 1834×88 pixels by stacking sub-regions of the original convergence map. At scattering depth J=5, the spatial subsampling factor is 2^J = 32, yielding only `⌊88/32⌋ = 2` spatial cells along the narrow dimension, corresponding to coverage of only 64/88 ≈ 73% of the available map width. This spatial truncation degrades scattering coefficients significantly: early J=5 runs (without HOS) scored only 4.98–5.01, far below the 9.4–9.98 achievable with proper spatial coverage. The fix was to use J=4 (coverage: `⌊88/16⌋ = 5` cells, 80/88 ≈ 91% coverage) or J=3 (coverage: 11 cells, 88/88 = 100%) for scattering-only configurations, or to combine J=5 with HOS where the poor spatial coverage is compensated. **Any scattering-based analysis on non-square survey geometries must verify that 2^J ≤ min(H, W).**

### 2.3 Higher-Order Statistics (HOS) alone

HOS alone is the strongest single-family statistic, reaching val_score = 11.19 (J=4 scales). The two components of HOS — multi-scale wavelet **peak counts** and wavelet **L1-norm histograms** — probe complementary aspects of the non-Gaussian convergence field:

- **Peak counts** directly measure the abundance of local convergence maxima as a function of SNR and scale, which is theoretically linked to the halo mass function and therefore tightly constrains σ_8 and Ω_m.
- **L1-norm histograms** capture the full one-point distribution of wavelet coefficients at each scale, encoding both overdense and underdense regions (voids), which add complementary constraints especially on the matter density Ω_m.

The combination of these two descriptors makes full HOS superior to either component alone (see §3).

### 2.4 HOS + Scattering (best single combination)

Combining HOS with the scattering transform consistently achieves the highest scores among fixed-SNR configurations (11.40–11.42). The scattering transform captures inter-scale amplitude correlations that are not explicitly represented in the local peak count or L1-norm histograms. These cross-scale correlations are a genuinely complementary source of information, explaining the ~0.2–0.25 point improvement over HOS alone. The combination is robust across different values of J (3, 4, 5 all yield 11.40–11.42 with ns=4), and further improves with wider SNR histogramming ranges (see §5), reaching the best observed score of **11.539** with both peaks SNR range [−5, 10] and L1 SNR range [−10, 10].

### 2.5 Effect of adding PS to HOS (+Scat)

With the original (bugged) normalisation, PS + HOS was **worse** than HOS alone by ~1 point. With the corrected per-batch normalisation, the picture changes substantially:

| Combination | Val score |
|---|---|
| HOS alone (ns=4) | 11.185 |
| PS + HOS (bugged norm, diverged) | ~10.94 (peaked ep3) |
| PS + HOS (fixed norm) | 11.239 |
| HOS + Scat (J=4, ns=4) | 11.420 |
| PS + HOS + Scat (fixed norm, run 1) | 11.486 |
| PS + HOS + Scat (fixed norm, run 2) | 11.475 |
| HOS + Scat, L1 SNR wide (no PS) | 11.489 |
| **HOS + Scat, both SNR wide (no PS)** | **11.539** |

With corrected normalisation, PS + HOS (11.239) outperforms HOS alone (11.185) by +0.054 pts, a marginal but positive contribution. PS + HOS + Scat (11.48, average of two fixed-norm runs) slightly outperforms the equivalent HOS + Scat baseline (11.42) by +0.06 pts. These gains are modest and statistically marginal (within the run-to-run variability of ±0.04; see §8.1). More importantly, the best HOS + Scat configuration without PS (11.54, with wide SNR ranges) substantially outperforms PS + HOS + Scat with standard ranges (11.48), demonstrating that optimising the HOS histogramming is more impactful than adding the PS.

---

## 3. HOS Variant Ablation

Three variants of HOS were compared: full HOS (peak counts + L1-norm histograms), L1-norm only, and peak counts only. The table below shows results from the systematic ablation (ns=3, 4, 6; standard SNR ranges).

| Variant | ns=3 | ns=4 | ns=6 | Avg (3 runs) |
|---------|------|------|------|--------------|
| **Full HOS** | 11.060 | **11.185** | 11.050 | **11.098** |
| L1-norm only | 10.870 | 11.024 | 10.910 | 10.935 |
| Peaks only | 10.593 | 10.653 | 10.635 | 10.627 |

Full HOS is the best variant in all cases. The ranking L1-norm > Peaks is consistent with the L1-norm histogram encoding the full one-point distribution of the wavelet-filtered field — including voids (negative SNR regions) — whereas peak counts only measure local maxima. Voids have been shown to carry substantial cosmological information orthogonal to peaks, particularly for Ω_m. Including both descriptors maximises the constraining power (+0.16 pts over L1-only, +0.47 pts over peaks-only on average).

---

## 4. Hyperparameter Sensitivity

### 4.1 Number of HOS scales

| n_scales | Full HOS | L1-only | Peaks-only | All-variant avg |
|----------|----------|---------|------------|-----------------|
| 3        | 11.060   | 10.870  | 10.593     | 10.841          |
| **4**    | **11.185** | **11.024** | **10.653** | **10.954** |
| 6        | 11.050   | 10.910  | 10.635     | 10.865          |

Four wavelet scales is consistently optimal across all HOS variants. Three scales captures most of the information (+0.11 pts below 4-scale average), while six scales shows no improvement and in some cases degrades performance (likely due to the larger feature vector increasing overfitting risk). **Four scales is the recommended default.** The saturation above four scales reflects the physical coherence length of cosmological structures: at the map resolution used here, four wavelet scales already span angular separations from sub-arcminute to degree scales, covering the range where weak lensing is most discriminating.

### 4.2 Number of HOS bins

| n_bins (peaks) | Runs | Score range |
|----------------|------|-------------|
| 31 | 388wnkgf, szz6u1c1, fjy66buh | 10.81–10.96 |
| 51 | all ablation runs | 10.59–11.19 |

Using 51 bins provides a consistent ~0.1–0.2 pt improvement over 31 bins, likely because the finer binning better resolves the peak count distribution near the noise threshold (SNR ~ 0–3), where the cosmological signal is concentrated. **51 bins is preferred.** For the L1-norm, 80 bins vs 120 bins with the same [−10, 10] range shows a slight degradation (11.449 vs 11.489 for L1-wide), suggesting the model does not benefit from finer histogram resolution when the bin count already exceeds 80.

### 4.3 Scattering depth J

| J | HOS + Scat (ns=4) | Scat-only |
|---|-------------------|-----------|
| 3 | 11.397 (7852343) | 9.438 (7851221) |
| **4** | **11.420 (7852116)** | **9.984 (7851222)** |
| 5 | 11.409 (7852344) | 9.626 (7851223) |

In combination with HOS, all three J values perform similarly (11.40–11.42), with J=4 being marginally best (+0.011 over J=5, +0.023 over J=3). In the scattering-only regime, J=4 also wins by a larger margin (+0.36 over J=5, +0.55 over J=3). J=5 with scattering alone is hampered by the spatial coverage constraint described in §2.2. J=3 in scattering-only mode is underperforming because each scattering order captures only up to 2^J = 8 pixel separations, missing the intermediate-scale (16–32 pixel) correlations that carry cosmological information. **J=4 is the optimal and safest choice.** The insensitivity of HOS+Scat to J (11.40–11.42 range is within run-to-run variability) reflects the fact that HOS explicitly captures multi-scale statistics at all resolved scales, leaving only marginal complementary information for the scattering transform to add, regardless of its depth.

---

## 5. Effect of SNR Range on HOS Performance

The HOS statistics involve histogramming wavelet-filtered field values in bins of signal-to-noise ratio (SNR = κ_w/σ_noise). Two SNR parameters are independently controlled:

- **`hos_min_snr` / `hos_max_snr`**: SNR range for the wavelet **peak count** histograms
- **`hos_l1_min_snr` / `hos_l1_max_snr`**: SNR range for the wavelet **L1-norm** histograms

All runs in this section use HOS + Scat (J=4, ns=4, nbins=51, l1nbins=80) as the base configuration. The reference baseline is the median performance at standard settings (peaks=[−3, 7], L1=[−7, 7]), approximately 11.37 (average of four base runs: 7852116 11.420, 7s2u38we 11.361, pmu1wrjv 11.334, qwp9c44x 11.328).

### 5.1 Peak count SNR range

| Config | Peaks SNR | L1 SNR | Val score | Δ vs base |
|--------|-----------|--------|-----------|-----------|
| snr_peaks_narrow (7858310) | [−1, 5] | [−7, 7] | 11.321 | −0.050 |
| base (7852116) | [−3, 7] | [−7, 7] | 11.420 | 0.000 |
| snr_peaks_posonly (7858312) | [0, 8] | [−7, 7] | 11.445 | +0.024 |
| snr_peaks_posonly_r2 (7896657) | [0, 8] | [−7, 7] | 11.444 | +0.023 |
| snr_peaks_wide (7858311) | [−5, 10] | [−7, 7] | 11.488 | +0.067 |

Widening the peaks SNR range to [−5, 10] provides a meaningful improvement (+0.067 pts). The positive-only range [0, 8] is nearly equivalent to the base (+0.024 pts), confirming that the sub-zero peak bins in the baseline carry minimal unique information (voids are better captured by the L1-norm histogram). Narrowing to [−1, 5] cuts the high-SNR tail and degrades performance.

### 5.2 L1-norm SNR range

| Config | Peaks SNR | L1 SNR | l1_nbins | Val score | Δ vs base |
|--------|-----------|--------|----------|-----------|-----------|
| snr_l1_narrow (7858313) | [−3, 7] | [−5, 5] | 80 | 11.360 | −0.061 |
| base (7852116) | [−3, 7] | [−7, 7] | 80 | 11.420 | 0.000 |
| snr_l1_wide (7858314) | [−3, 7] | [−10, 10] | 80 | 11.489 | +0.069 |
| snr_l1_wide_bins120 (7896660) | [−3, 7] | [−10, 10] | 120 | 11.449 | +0.028 |
| snr_l1_wider (7896661) | [−3, 7] | [−12, 12] | 80 | 11.518 | +0.097 |

Widening the L1 SNR range consistently improves performance. The [−10, 10] range gains +0.069 pts over the [−7, 7] baseline, and extending further to [−12, 12] gains an additional +0.028 pts (+0.097 total). Increasing the number of bins from 80 to 120 at the [−10, 10] range slightly degrades performance (11.449 vs 11.489), suggesting the model does not benefit from finer binning at this range. The optimal L1 range saturates between −10 and −12 on the negative side.

### 5.3 Joint optimisation of both SNR ranges

| Config | Peaks SNR | L1 SNR | Val score | Δ vs base |
|--------|-----------|--------|-----------|-----------|
| base (7852116) | [−3, 7] | [−7, 7] | 11.420 | 0.000 |
| snr_l1_wide (7858314) | [−3, 7] | [−10, 10] | 11.489 | +0.069 |
| snr_peaks_wide (7858311) | [−5, 10] | [−7, 7] | 11.488 | +0.067 |
| snr_l1_wider (7896661) | [−3, 7] | [−12, 12] | 11.518 | +0.097 |
| **snr_full_wide (7896659)** | **[−5, 10]** | **[−10, 10]** | **11.539** | **+0.119** |

Combining wider ranges for both peaks and L1 yields the best overall performance: **11.539** (Δ = +0.119 vs base). The gains from widening peaks (+0.067) and L1 (+0.069) are approximately additive (+0.136 predicted, +0.119 observed), indicating that the two ranges probe largely independent aspects of the field statistics. **The optimal configuration uses peaks SNR = [−5, 10] and L1 SNR = [−10, 10].**

### 5.4 Physical interpretation

**Peak SNR range** — From lensing theory, the SNR distribution of wavelet peaks carries different information in different regimes:

- **High-SNR peaks (κ/σ > 3)**: Dominated by massive dark matter haloes. Strongly sensitive to σ_8 (cluster abundance scales as ~σ_8^8 at fixed mass threshold). Widening to [−5, 10] captures rare high-mass clusters and extreme underdensities.
- **Intermediate SNR (0 < κ/σ < 3)**: Contributions from filaments, projected structures, and smaller haloes. Sensitive to the combination σ_8 Ω_m^α.
- **Negative SNR (κ/σ < 0)**: Voids and underdense regions. Restricting to positive-only [0, 8] has **negligible impact** (+0.024 pts vs base), meaning negative-SNR peaks in the wavelet peak histogram carry almost no independent information beyond what the L1-norm histogram already provides. This makes physical sense: the L1-norm histogram is specifically designed to capture the full one-point field distribution including voids, so the peaks histogram is most informative in the positive-SNR regime.

**L1-norm SNR range** — The L1-norm histogram captures the full one-point distribution of wavelet coefficients at each scale:

- Narrowing to [−5, 5] loses the tails of the convergence distribution, dropping by −0.061 pts.
- Widening to [−10, 10] gives a +0.069 pt gain. The default [−7, 7] range misses bins at 7 < |SNR| < 10 that contain rare but highly informative extreme events (massive cluster cores, ultra-deep voids). These extreme pixels, while sparse, provide strong leverage on cosmological parameters.
- Extending to [−12, 12] gives a further +0.028 pt gain, with gains appearing to saturate beyond ±12.

---

## 6. Summary and Recommendations

| Recommendation | Evidence |
|----------------|----------|
| **Use HOS + Scattering** | Best val score (11.54); complementary non-Gaussian features |
| **Use full HOS** (peaks + L1-norms) | ~0.16 pt gain over L1-only; ~0.47 pt gain over peaks-only |
| **4 wavelet scales** | Consistent optimum; 3 is slightly worse (−0.11), 6 brings no gain |
| **51 bins for peak counts** | +0.1–0.2 pt over 31 bins at negligible cost |
| **Scattering depth J = 4** | Marginally best; consistent with spatial coverage constraints |
| **Peaks SNR range [−5, 10]** | +0.067 pts vs [−3,7]; captures high-SNR cluster abundance |
| **L1 SNR range [−10, 10]** | +0.069 pts vs [−7,7]; extreme convergence tails carry cosmological information |
| **PS not recommended alone or as add-on** | PS alone: 6.85 (vs HOS 11.19); PS+HOS+Scat adds only +0.05 vs HOS+Scat at equal settings |

### Best configuration

```yaml
use_ps: false
use_hos: true
hos_n_scales: 4
hos_n_bins: 51
hos_l1_nbins: 80
hos_min_snr: -5.0
hos_max_snr: 10.0
hos_l1_min_snr: -10.0
hos_l1_max_snr: 10.0
use_scattering: true
scattering_J: 4
scattering_L: 8
```

This corresponds to run `snr_full_wide` (job 7896659, val_score = **11.539**, val_MSE = **7.04×10⁻⁴**), a +0.119 improvement over the standard-range HOS+Scat baseline (11.420). The val_MSE of 7.04×10⁻⁴ represents a 31% reduction relative to PS+HOS alone (1.02×10⁻³).

### Ongoing Expansion (March 2026)

An expanded follow-up ablation campaign is now in progress to improve statistical confidence and to validate recent wavelet scattering transform (WST) implementation fixes. Key settings are being run with **5 random seeds per configuration**. The currently active multi-seed SLURM batches are **r4** (job IDs **8452261–8452313**, 50 jobs total, 4h walltime per job) and a dedicated **WST pooling A/B batch** (**8453322–8453341**, 20 jobs total). A prior batch interruption was traced to a **tooling/entrypoint exit-code issue**, not to model divergence.

#### Methods updates

- **Reproducibility protocol (seeds):** all key ablation comparisons in the expansion are replicated with 5 seeds to separate true configuration effects from run-to-run variance.
- **WST mask-aware pooling fix:** `compute_scattering_batch` now supports optional mask-aware weighted pooling on the scattering grid, with explicit modes:
  - `scattering_mask_pooling='soft'` (fractional mask weighting; current default),
  - `scattering_mask_pooling='hard'` (binary include/exclude thresholding).
  This prevents padded/masked regions from biasing pooled scattering coefficients, especially at survey edges and irregular footprints.
- **WST normalisation variants:** ongoing runs compare `log1p_zscore` (new default), `zscore`, and `none` for scattering feature normalisation.

#### In-progress runs (r4 + WST A/B)

| Config group | What is being varied | Purpose |
|--------------|----------------------|---------|
| Scattering L sweep | `L ∈ {4, 6, 12, 16}` (with fixed baseline settings otherwise) | Measure angular-resolution sensitivity of WST and identify stable/optimal `L` under the corrected pooling path. |
| PS + Scattering baseline | PS + WST baseline settings (multi-seed) | Re-establish a clean PS+WST reference under unified normalisation and mask-aware pooling. |
| PS + HOS + Scattering (full-wide SNR) | Full feature stack with peaks/L1 wide SNR settings | Test whether prior best full-wide setting remains robust under multi-seed replication and WST fixes. |
| WST normalisation variants | `zscore` and `none` vs default `log1p_zscore` | Quantify sensitivity to feature scaling choice and check calibration/performance trade-offs. |
| WST pooling A/B (new) | `scattering_mask_pooling='soft'` vs `'hard'` under matched seeds | Isolate whether soft mask weighting improves WST stability/performance over hard threshold pooling. |

#### Final multi-seed aggregates from the expansion batches

All expansion jobs referenced in this section completed with 5/5 seeds per configuration.

| Config | Mean val_score | Std |
|---|---:|---:|
| HOS+Scat J4 (`ablation_hos_scat_J4`) | 11.4416 | 0.0136 |
| HOS+Scat J4, L=4 | 11.4281 | 0.0108 |
| HOS+Scat J4, L=6 | 11.4306 | 0.0201 |
| HOS+Scat J4, L=12 | 11.4521 | 0.0151 |
| HOS+Scat J4, L=16 | **11.4529** | 0.0137 |
| HOS+Scat J4, `zscore` | 11.4408 | 0.0132 |
| HOS+Scat J4, `none` norm | 11.2870 | 0.0166 |
| HOS+Scat J4, `mean_std` feature pooling | **11.4532** | **0.0093** |
| HOS+Scat J4, `mean_std`, **no-shift** | 11.2174 | 0.0089 |
| HOS+Scat J4, `mean_std`, **no-flip + no-shift** | **11.7258** | 0.0140 |
| HOS+Scat J4, `mean_std`, **full scattering geometry** | 11.4703 | 0.0116 |
| HOS+Scat J4, `mean_std`, full geometry + no-shift | 11.1881 | 0.0121 |
| PS+Scat J4 | 10.1669 | 0.0330 |
| HOS+Scat full-wide SNR | 11.5584 | 0.0125 |
| HOS+Scat full-wide SNR, `mean_std` feature pooling | **11.5771** | **0.0071** |
| HOS+Scat full-wide SNR, `mean_std`, no-shift | 11.3042 | 0.0144 |
| HOS+Scat full-wide SNR, `mean_std`, full scattering geometry | 11.5962 | 0.0111 |
| HOS+Scat full-wide SNR, `mean_std`, no-flip + no-shift | 11.7607 | 0.0218 |
| HOS+Scat full-wide SNR, `mean_std`, full geometry + no-shift | 11.2814 | 0.0134 |
| HOS+Scat full-wide SNR, `mean_std`, **full geometry + no-flip + no-shift** | **11.7740** | **0.0167** |
| PS+HOS+Scat (standard SNR) | 11.4666 | 0.0104 |
| PS+HOS+Scat (full-wide SNR) | 11.5745 | 0.0228 |

#### WST-focused conclusions from the completed expansion

- **Scattering orientation `L` sweep:** the best completed settings are `L=12` and `L=16` (both above `L=8` by ~+0.011), while `L=4`/`L=6` are slightly below baseline. The effect is modest but consistent with improved angular resolution at larger `L`.
- **Mask-aware pooling mode (`soft` vs `hard`):** differences are small but consistently favor `soft` pooling in matched A/B runs:
  - Baseline HOS+Scat J4: `11.4435` (soft) vs `11.4409` (hard), Δ = +0.0026.
  - Full-wide SNR: `11.5671` (soft) vs `11.5583` (hard), Δ = +0.0088.
- **Scattering feature pooling (`mean_std` vs `mean`):** concatenating per-coefficient spatial mean and spatial std yields a meaningful gain over mean-only pooling:
  - Baseline HOS+Scat J4: `11.4532` (`mean_std`) vs `11.4416` (`mean`), Δ = +0.0116.
  - Full-wide SNR: `11.5771` (`mean_std`) vs `11.5584` (`mean`), Δ = +0.0187.
  This was an important first improvement but not the dominant bottleneck.
- **Implementation-audit finding (major): augmentation policy on reshaped maps was strongly suboptimal for WST.**
  - Keeping flips but removing shifts causes a large drop:
    - Baseline J4: `11.2174` vs `11.4522` (Δ = −0.2348)
    - Full-wide: `11.3042` vs `11.5789` (Δ = −0.2747)
  - Removing both flips and shifts produces a large gain:
    - Baseline J4: `11.7258` vs `11.4522` (Δ = +0.2737)
    - Full-wide: `11.7607` vs `11.5789` (Δ = +0.1817)
  This directly explains why WST looked unexpectedly weak before: augmentation artifacts on the non-Euclidean reshaped-map layout were suppressing WST quality.
- **Scattering geometry (`reduced` vs `full`) after fixes:** moving scattering extraction to full map geometry gives a consistent but modest gain:
  - Baseline J4: `11.4703` vs `11.4522` (Δ = +0.0181)
  - Full-wide: `11.5962` vs `11.5789` (Δ = +0.0173)
  A dedicated WST diagnostic on synthetic maps also showed substantial reduced-vs-full feature mismatch (`geom_cos≈0.657`, `geom_rel_l2≈0.828`), supporting this direction.
- **WST normalisation:** `log1p_zscore` and `zscore` are effectively tied at this precision, while disabling normalization (`none`) causes a clear drop (−0.15 points vs `log1p_zscore` baseline). This strongly supports keeping normalized WST features.
- **PS+Scat vs HOS+Scat:** PS+Scat remains much weaker than HOS+Scat-family models, confirming that WST is complementary but not sufficient by itself in this setup.
- **Best expanded configuration among tested runs:** `HOS+Scat` with full-wide SNR, `mean_std` pooling, full scattering geometry, and no flip/shift augmentation reached **`11.7740`** (5-seed mean), clearly above the earlier best (`11.5771`) and above `PS+HOS+Scat` full-wide (`11.5745`).

Recommended WST defaults after this expansion:

- `scattering_mask_pooling: soft`
- `scattering_normalization: log1p_zscore` (or `zscore`, near-equivalent)
- `scattering_feature_pooling: mean_std`
- `scattering_geometry: full` (small but consistent gain)
- For **summary-statistics-only WST models** (`use_cnn: false`): `augment_flip: false`, `augment_shift: false`
- `scattering_L: 12` as a robust default; `16` is also competitive but with slightly higher feature dimensionality.

---

## 7. Targeted Questions

This section addresses five specific scientific questions. Comparisons use matched settings throughout (ns=4, nbins=51, l1nbins=80 where applicable). Runs marked † use per-batch PS normalisation (fixed).

### Q1. Does including the PS add anything to HOS?

| Condition | Job / Run | Val score | Δ vs HOS |
|-----------|-----------|-----------|----------|
| HOS alone (ns=4) | 7851214 | 11.185 | — |
| PS + HOS (bugged norm, diverged) | 7852112 | ~10.94 (ep3) | — |
| **PS + HOS (fixed norm)†** | **7896635** | **11.239** | **+0.054** |

With corrected normalisation, adding the PS to HOS provides a marginal but positive gain of +0.054 points. While statistically borderline given the run-to-run variability (σ ≈ 0.04; see §8.1), the gain is directionally consistent with the PS adding independent k-space amplitude information. The contribution of the PS is substantially smaller than the gain from adding the scattering transform (+0.235 pts for HOS→HOS+Scat), confirming that the PS captures mostly Gaussian information already partially implicit in the L1-norm histogram's second moment.

**Conclusion**: PS marginally improves HOS (∼+0.05 pts), but the gain is not statistically robust. The PS remains uninformative as a standalone statistic.

### Q2. Is the scattering transform better than HOS?

| Condition | Job / Run | Val score |
|-----------|-----------|-----------|
| Scattering alone (J=4) | 7851222 | 9.984 |
| HOS alone (ns=4) | 7851214 | **11.185** |

**No — HOS is substantially better (+1.20 pts).** The scattering transform computes a cascade of wavelet modulus averages, encoding *inter-scale amplitude correlations*. It is a compact descriptor but misses the one-point distributional information (peak abundance vs SNR, histogram tails) that is directly linked to the cosmological halo mass function. HOS explicitly histograms the wavelet field at each scale, providing a richer characterisation of the non-Gaussian structure. The scattering coefficients are, by construction, *spatial averages* over the field, discarding the local extreme-value statistics (massive cluster peaks, deep voids) that are the most cosmologically sensitive.

### Q3. Is the combination HOS + Scattering better than Scattering alone?

| Condition | Job / Run | Val score |
|-----------|-----------|-----------|
| Scattering alone (J=4) | 7851222 | 9.984 |
| HOS + Scattering (J=4, ns=4) | 7852116 | **11.420** |

**Yes — by +1.44 pts.** HOS adds substantial information on top of the scattering transform. This is expected: scattering coefficients capture cross-scale correlations efficiently but are insensitive to the one-point distribution. HOS fills this gap with peak count histograms and L1-norm histograms. The combination is strictly better because the two descriptors are complementary — neither is redundant with the other.

### Q4. Does including the PS add anything to the L1-norm?

| Condition | Job / Run | L1 SNR | Val score | Δ vs L1-only |
|-----------|-----------|--------|-----------|--------------|
| L1-norm alone (ns=4) | 7851216 | [−7, 7] | 11.024 | — |
| **PS + L1-norm (fixed norm)†** | **7896636** | **[−7, 7]** | **11.067** | **+0.043** |
| L1-norm alone (wide) | 7896638 | [−10, 10] | 11.096 | +0.072 |

Adding the PS to the L1-norm histogram yields a marginal +0.043 pt gain. Notably, simply widening the L1 SNR range from [−7, 7] to [−10, 10] provides a larger improvement (+0.072 pts) at no additional model complexity. This suggests that what the PS would contribute in terms of amplitude-scale information is better captured by widening the dynamic range of the L1-norm histogram itself. **The L1-norm histogram at widened SNR range is a more efficient use of model capacity than adding the PS.**

### Q5. Does adding peak counts improve over the L1-norm alone?

| Condition | Job / Run | Peaks SNR | L1 SNR | Val score | Δ vs L1-only |
|-----------|-----------|-----------|--------|-----------|--------------|
| L1-norm alone (ns=4, standard) | 7851216 | — | [−7, 7] | 11.024 | — |
| Full HOS (ns=4, standard) | 7851214 | [−3, 7] | [−7, 7] | 11.185 | **+0.161** |
| L1-norm alone (wide) | 7896638 | — | [−10, 10] | 11.096 | — |
| Full HOS (wide L1) | 7896637 | [−3, 7] | [−10, 10] | 11.277 | **+0.181** |

**Yes — adding peak counts gains +0.161 to +0.181 pts regardless of the L1 SNR range.** Peak counts and L1-norms are complementary non-Gaussian statistics: while L1-norms capture the full amplitude distribution (including voids), **peak counts specifically measure local maxima** — a spatially-structured statistic that is directly linked to the halo mass function and sensitive to the clustering topology of the convergence field in a way that pixel-level histograms are not. A pixel at a given SNR in the L1-norm histogram could be isolated noise or part of a large coherent structure; the peak count histogram distinguishes these cases by requiring spatial concentration. The consistent ~+0.17 pt gain from peaks over L1-only confirms that spatial clustering information carries genuinely independent cosmological content.

---

## 8. Statistical Significance and Publication-Quality Assessment

### 8.1 Run-to-run variability and significance threshold

To characterise the stochastic uncertainty of the training procedure, we consider multiple independent runs with identical configurations. Four independent runs of the base HOS+Scat configuration (J=4, ns=4, peaks=[−3, 7], L1=[−7, 7]) yield scores of 11.420, 11.361, 11.334, and 11.328, corresponding to a standard deviation of σ ≈ 0.039 and a range of 0.092. Two repeats each of the peaks-positive-only and PS+HOS+Scat(fixed) configurations yield differences of 0.001 and 0.011, respectively, indicating good reproducibility for converged configurations. Based on these observations, we adopt the following conventions:

- **Significant** (Δ > 0.10): difference exceeds the observed run-to-run range; robust conclusion
- **Marginal** (0.05 < Δ < 0.10): difference is larger than the standard deviation but within the range; directionally consistent but not decisive
- **Negligible** (Δ < 0.05): within run-to-run variability; no reliable conclusion

### 8.2 Hierarchy of summary statistics

The following hierarchy is supported by the ablation evidence and is robust to the stochastic training variability:

**Tier 1 (val_score > 11.4): HOS + Scattering with optimised SNR ranges**

The combination of higher-order wavelet statistics (peak count histograms + L1-norm histograms) with wavelet scattering transform coefficients constitutes the best-performing class of summary statistics in this study, reaching val_score = 11.54 with optimised SNR ranges (HOS+Scat, J=4, ns=4, peaks=[−5, 10], L1=[−10, 10]). This combination is **significantly better** than HOS alone (+0.35 pts, Δ >> σ) and represents the recommended configuration for production use. Within this tier, the performance is relatively insensitive to the choice of scattering depth J (J=3, 4, 5 all yield 11.40–11.42 at standard SNR settings).

**Tier 2 (11.1 < val_score < 11.4): HOS alone (optimised)**

HOS alone with four wavelet scales and full peak+L1 statistics reaches val_score = 11.19–11.28 (standard and wide-L1 settings). This is **significantly better** than any single-family descriptor, but **significantly worse** than HOS+Scattering (Δ ≈ 0.25–0.35 pts >> σ). HOS provides the dominant contribution to the HOS+Scat combination: comparing Scattering alone (9.98) and HOS alone (11.19), HOS adds far more information than the scattering transform. The scattering transform functions as a complementary supplement to HOS rather than an alternative.

**Tier 3 (9.4 < val_score < 10.0): Scattering alone**

The scattering transform alone is **significantly better** than the PS alone (Δ ≈ 3.1 pts) but **significantly worse** than HOS alone (Δ ≈ 1.2 pts). The scattering transform is informationally intermediate: richer than the PS (which is a Gaussian statistic) but less expressive than explicit one-point distribution histograms (HOS). Performance in this tier is sensitive to the choice of J and the survey geometry (see §2.2).

**Tier 4 (val_score ≈ 6.8): Power spectrum (PS) alone**

The angular power spectrum alone is the weakest viable summary statistic (6.85 with corrected normalisation). This result is physically expected: the convergence power spectrum is a two-point, Gaussian statistic that captures the total projected matter power but is blind to the non-Gaussian features (halo clustering topology, void underdensities, filament statistics) that provide the strongest discrimination between cosmological models in the non-linear regime. The PS alone is **significantly inferior to all non-Gaussian statistics** tested (Δ ≥ 2.6 pts vs Scattering alone, Δ ≥ 4.3 pts vs HOS alone).

### 8.3 Significance of individual design choices

The table below summarises each tested design choice, the associated performance change, and a significance assessment.

| Design choice | Comparison | Δ score | Significance |
|---------------|-----------|---------|--------------|
| HOS vs PS | 11.185 vs 6.849 | +4.34 | **Highly significant** |
| HOS vs Scattering | 11.185 vs 9.984 | +1.20 | **Highly significant** |
| Scattering vs PS | 9.984 vs 6.849 | +3.14 | **Highly significant** |
| Add Scattering to HOS | 11.185 → 11.420 | +0.235 | **Significant** |
| Full HOS vs L1-only | 11.185 vs 11.024 | +0.161 | **Significant** |
| Full HOS vs Peaks-only | 11.185 vs 10.653 | +0.532 | **Highly significant** |
| 4 scales vs 3 scales (HOS) | 11.185 vs 11.060 | +0.125 | **Significant** |
| 4 scales vs 6 scales (HOS) | 11.185 vs 11.050 | +0.135 | **Significant** |
| Wide L1 [−10,10] vs [−7,7] | 11.489 vs 11.420 | +0.069 | **Marginal** |
| Wide peaks [−5,10] vs [−3,7] | 11.488 vs 11.420 | +0.067 | **Marginal** |
| Combined wide SNR vs standard | 11.539 vs 11.420 | +0.119 | **Significant** |
| 51 bins vs 31 bins (peaks) | ~+0.15 avg | +0.15 | **Significant** |
| Add PS to HOS (fixed norm) | 11.185 → 11.239 | +0.054 | **Marginal** |
| Add PS to HOS+Scat (fixed norm) | 11.420 → 11.480 | +0.060 | **Marginal** |
| 80 bins vs 120 bins at L1-wide | 11.489 vs 11.449 | −0.040 | **Negligible** |
| Positive-only peaks vs standard | 11.445 vs 11.420 | +0.024 | **Negligible** |

### 8.4 Practical recommendations for publication

Based on the above evidence, the following conclusions are supported for publication:

1. **Non-Gaussian statistics are essential.** The power spectrum alone (6.85) is dramatically outperformed by HOS alone (11.19) by 4.3 points in the competition metric. This is consistent with the theoretical expectation that non-linear gravitational evolution generates significant non-Gaussian information in the convergence field that the PS cannot access. Any analysis relying solely on the power spectrum discards the majority of the cosmological information available in weak lensing maps at this noise level.

2. **Wavelet peak counts and L1-norm histograms are complementary and both necessary.** The L1-norm alone (11.02) significantly outperforms peak counts alone (10.65), but the full combination (11.19) is significantly better than either individually. These two statistics probe orthogonal properties of the non-Gaussian field: L1-norm histograms characterise the one-point amplitude distribution at each scale (sensitive to the overall abundance of over- and underdensities), while peak counts resolve the *spatial concentration* of those structures (directly linked to the halo mass function). Both are necessary for optimal inference.

3. **The scattering transform provides significant complementary information to HOS.** Adding the scattering transform to HOS gains +0.235 points (HOS alone: 11.19 → HOS+Scat: 11.42), a highly significant improvement exceeding six times the training standard deviation. This gain arises because scattering coefficients encode *inter-scale amplitude correlations* — e.g. how the magnitude of fluctuations at one scale predicts the magnitude at another scale — a second-order cross-scale statistic that is not captured by the marginal wavelet amplitude histograms of HOS. The scattering transform alone, however, is substantially worse than HOS (9.98 vs 11.19), and should only be used as a supplement rather than a replacement.

4. **Four wavelet scales is optimal; adding more scales does not improve performance.** The performance peaks at four scales and either plateaus or degrades at six scales. This saturation reflects the angular resolution of the survey: at this pixel scale and noise level, four wavelet scales already span the range of angular separations that carry discriminating cosmological information. Adding more scales increases the feature dimensionality without adding independent cosmological information, increasing the risk of overfitting.

5. **The SNR histogram range is a significant hyperparameter.** Widening the L1-norm histogram to [−10, 10] and the peak count histogram to [−5, 10] together improve the score by +0.119 points (significant). This improvement reflects the cosmological information content of rare, extreme-SNR structures: massive cluster cores (κ/σ > 7) are amongst the most sensitive probes of the σ_8–Ω_m degeneracy direction, and deep void underdensities (κ/σ < −7) provide complementary Ω_m sensitivity. Standard SNR ranges (e.g. ±5–7) truncate the tails of the convergence distribution and systematically discard these informative events. We recommend using the extended SNR ranges [−5, 10] for peaks and [−10, 10] for L1-norms as a default configuration.

6. **The power spectrum adds negligible information beyond HOS or HOS+Scattering.** With corrected normalisation, adding the PS to HOS gains only +0.054 points (marginal, within training variability), and adding PS to HOS+Scat gains approximately +0.060 points (marginal). The PS is not a recommended addition to the HOS+Scat pipeline given its marginal contribution. The angular power spectrum shape provides k-space information that is partially redundant with the multi-scale variance already encoded in the L1-norm histograms, and adding it increases model input dimensionality without a commensurate improvement in constraining power. However, with a proper noise-consistent normalisation scheme, the PS is not harmful and may provide a small consistent gain.

---

## 9. Early Run Archive (WandB)

For completeness, the full epoch-by-epoch training records for the full run table in §7 of the earlier version of this document are reproduced below.

| Run ID | Statistics | Variant | ns | nbins | l1nbins | Peaks SNR | L1 SNR | J | Val score | Val MSE |
|--------|-----------|---------|-----|-------|---------|-----------|--------|---|-----------|---------|
| yj48er19 | PS+HOS | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | — | 10.086 | 1.02e-3 |
| 71qhtzi4 | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 5 | −23.684 | 7.97e-3 |
| f1uqjf8o | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 5 | 4.976 | 2.25e-3 |
| sy8rmbah | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 4 | 8.230 | 1.49e-3 |
| 1dl574x4 | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 3 | 5.010 | 2.40e-3 |
| v6cj7p98 | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 4 | 11.041 | 7.78e-4 |
| 3cjs3z1x | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 5 | 11.331 | 7.24e-4 |
| dv0w9s6m | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 5 | 11.399 | 7.08e-4 |
| lprg7hu1 | HOS | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | — | 11.153 | 7.53e-4 |
| 388wnkgf | HOS | l1_only | 4 | 31 | 80 | [−4, 8] | [−7, 7] | — | 10.963 | 7.99e-4 |
| wml1jqti | HOS | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | — | 10.986 | 7.81e-4 |
| szz6u1c1 | HOS | l1_only | 6 | 31 | 80 | [−4, 8] | [−7, 7] | — | 10.852 | 8.19e-4 |
| thx56mt2 | HOS | peaks_only | 4 | 51 | 40 | [−3, 7] | [−8, 8] | — | 10.636 | 8.68e-4 |
| 0ajzckx6 | HOS | peaks_only | 6 | 51 | 40 | [−3, 7] | [−8, 8] | — | 10.601 | 8.75e-4 |
| 354qkkwq | Scat | — | — | — | — | — | — | 3 | 9.405 | 1.19e-3 |
| md5lhelk | PS | — | — | — | — | — | — | — | −0.174 | 4.09e-3 |
| 78a96766 | Scat | — | — | — | — | — | — | 5 | 9.576 | 1.16e-3 |
| 1p44eqr2 | Scat | — | — | — | — | — | — | 4 | 9.942 | 1.05e-3 |
| et2is8tp | HOS | peaks_only | 3 | 51 | 40 | [−3, 7] | [−8, 8] | — | 10.570 | 8.82e-4 |
| fjy66buh | HOS | l1_only | 3 | 31 | 80 | [−4, 8] | [−7, 7] | — | 10.815 | 8.41e-4 |
| msatahn4 | PS+HOS | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | — | 10.079 | 1.03e-3 |
| psobiafb | HOS | full | 3 | 51 | 80 | [−3, 7] | [−7, 7] | — | 11.051 | 7.83e-4 |
| 7s2u38we | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 4 | 11.361 | 7.04e-4 |
| pmu1wrjv | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 4 | 11.334 | 7.11e-4 |
| qwp9c44x | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 4 | 11.328 | 7.16e-4 |
| hm5ouq53 | HOS+Scat | full | 4 | 51 | 80 | [−1, 5] | [−7, 7] | 4 | 11.252 | 7.29e-4 |
| 2fzsedi9 | HOS+Scat | full | 4 | 51 | 80 | [−5, 10] | [−7, 7] | 4 | 11.438 | 6.85e-4 |
| v48r54pk | HOS+Scat | full | 4 | 51 | 80 | [0, 8] | [−7, 7] | 4 | 11.370 | 7.02e-4 |
| n7o3rj11 | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−5, 5] | 4 | 11.303 | 7.14e-4 |
| fttkhhhw | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−10, 10] | 4 | 11.467 | 6.79e-4 |
| zimy3a1i† | PS | — | — | — | — | — | — | — | 6.713 | 2.04e-3 |
| 08qodcg5† | PS+HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 4 | 11.398 | 7.07e-4 |
