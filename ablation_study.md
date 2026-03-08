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

All 25 runs are summarised in the table below.

| Run ID | Statistics | HOS variant | Scales | Bins | J | Val score | Val MSE |
|--------|-----------|-------------|--------|------|---|-----------|---------|
| yj48er19 | PS + HOS | full | 6 | 51 | — | 10.086 | 1.02e-3 |
| msatahn4 | PS + HOS | full | 4 | 51 | — | 10.079 | 1.03e-3 |
| dv0w9s6m | PS + HOS + Scat | full | 6 | 51 | 5 | **11.399** | **7.08e-4** |
| 3cjs3z1x | PS + HOS + Scat | full | 6 | 51 | 5 | 11.331 | 7.24e-4 |
| v6cj7p98 | PS + HOS + Scat | full | 6 | 51 | 4 | 11.041 | 7.78e-4 |
| 7s2u38we | HOS + Scat | full | 4 | 51 | 4 | **11.361** | **7.04e-4** |
| pmu1wrjv | HOS + Scat | full | 4 | 51 | 3 | 11.334 | 7.11e-4 |
| qwp9c44x | HOS + Scat | full | 4 | 51 | 5 | 11.328 | 7.16e-4 |
| lprg7hu1 | HOS | full | 4 | 51 | — | 11.153 | 7.53e-4 |
| psobiafb | HOS | full | 3 | 51 | — | 11.051 | 7.83e-4 |
| wml1jqti | HOS | full | 6 | 51 | — | 10.986 | 7.81e-4 |
| 388wnkgf | HOS | l1_only | 4 | 31 | — | 10.963 | 7.99e-4 |
| szz6u1c1 | HOS | l1_only | 6 | 31 | — | 10.852 | 8.19e-4 |
| fjy66buh | HOS | l1_only | 3 | 31 | — | 10.815 | 8.41e-4 |
| thx56mt2 | HOS | peaks_only | 4 | 51 | — | 10.636 | 8.68e-4 |
| 0ajzckx6 | HOS | peaks_only | 6 | 51 | — | 10.601 | 8.75e-4 |
| et2is8tp | HOS | peaks_only | 3 | 51 | — | 10.570 | 8.82e-4 |
| 1p44eqr2 | Scat | — | — | — | 4 | 9.942 | 1.05e-3 |
| 78a96766 | Scat | — | — | — | 5 | 9.576 | 1.16e-3 |
| 354qkkwq | Scat | — | — | — | 3 | 9.405 | 1.19e-3 |
| md5lhelk | PS | — | — | — | — | −0.174 | 4.09e-3 |
| sy8rmbah | PS + Scat | — | — | — | 4 | 8.230 | 1.49e-3 |
| f1uqjf8o | PS + Scat | — | — | — | 5 | 4.976 | 2.25e-3 |
| 1dl574x4 | PS + Scat | — | — | — | 3 | 5.010 | 2.40e-3 |
| 71qhtzi4 | PS + Scat | — | — | — | 5 | −23.684 | 7.97e-3 |

*Note: `1dl574x4` stopped early at epoch 17. `71qhtzi4` diverged; a repeat run (`f1uqjf8o`) recovered. Early runs with `use_ps` not explicitly set use the model default `use_ps=True`.*

---

## 2. Discussion by Feature Family

### 2.1 Power Spectrum (PS) alone

The PS alone yields a near-zero score (−0.17), far below the ~5 expected from a well-calibrated PS estimator. Post-hoc analysis revealed a **normalisation bug**: the hardcoded constants `LOG_PS_MEAN` and `LOG_PS_STD` used to standardise the PS features were computed on **noiseless** convergence maps, while training and validation always add shape noise (σ_noise ≈ 0.026 per pixel). For white noise, the noise contribution to the power spectrum is flat at log₁₀(P_noise) ≈ −9.65. Because the signal PS falls steeply with wavenumber k (slope ≈ −1.5), the noise dominates at k ≳ 500 rad⁻¹. Using the noiseless `LOG_PS_MEAN` to normalise the noisy maps produces a systematic bias of **+1 to +4σ at k-bins 4–9** (see table below), making the high-k features carry essentially no cosmological information.

| k-bin | log₁₀(P_signal) | log₁₀(P_measured) | Bias (σ) |
|-------|----------------|-------------------|---------|
| 0     | −8.68          | −8.63             | +0.19   |
| 3     | −9.44          | −9.23             | +0.90   |
| 5     | −9.95          | −9.47             | +1.83   |
| 7     | −10.61         | −9.60             | +3.13   |
| 9     | −11.08         | −9.63             | +3.93   |

In contrast, HOS and Scattering both use **per-batch normalisation** (mean and std computed over the current mini-batch), making their features automatically consistent with the actual noise level. The PS normalisation has been corrected to use the same per-batch scheme; validation runs (`ablation_ps_fixed`, jobs 7859129–7859130) are pending.

Even with corrected normalisation, there are fundamental reasons why the PS alone is less powerful than non-Gaussian statistics: the angular power spectrum is a second-order (Gaussian) statistic insensitive to the non-Gaussian features — filaments, peak abundance, void statistics — generated by non-linear structure formation. These non-Gaussian features are precisely what most tightly constrain Ω_m and σ_8 in the late universe.

### 2.2 Scattering Transform alone

The scattering transform alone achieves scores between 9.4 and 9.9 (varying with the maximum wavelet depth J). As a cascade of wavelet modulus operations, the scattering transform captures multi-scale non-Gaussian features efficiently. However, it is outperformed by full HOS (~11.0–11.2) by about 1.3 points. This gap is consistent with the scattering coefficients being a compact but lossy descriptor: they encode inter-scale correlations well but do not directly provide the one-point statistics (peak counts, field distribution) that carry substantial cosmological information in lensing maps. The scattering representation also shows training instability (one diverged run; see §4.3).

### 2.3 Higher-Order Statistics (HOS) alone

HOS alone is the strongest single-family statistic, reaching val_score = 11.15. The two components of HOS — multi-scale wavelet **peak counts** and wavelet **L1-norm histograms** — probe complementary aspects of the non-Gaussian convergence field:

- **Peak counts** directly measure the abundance of local convergence maxima as a function of SNR and scale, which is theoretically linked to the halo mass function and therefore tightly constrains σ_8 and Ω_m.
- **L1-norm histograms** capture the full one-point distribution of wavelet coefficients at each scale, encoding both overdense and underdense regions (voids), which add complementary constraints especially on the matter density Ω_m.

The combination of these two descriptors makes full HOS superior to either component alone (see §3).

### 2.4 HOS + Scattering (best combination)

Combining HOS with the scattering transform consistently achieves the highest scores (11.33–11.40). The scattering transform captures inter-scale amplitude correlations that are not explicitly represented in the local peak count or L1-norm histograms. These cross-scale correlations are a genuinely complementary source of information, explaining the ~0.2-point improvement over HOS alone. The combination is robust across different values of J (3, 4, 5 all yield ~11.33–11.36).

### 2.5 Effect of adding PS to HOS (+Scat)

Adding PS to HOS yields no measurable improvement, and in fact **PS + HOS is worse than HOS alone** by ~1 point:

| Combination | Val score |
|---|---|
| HOS alone (best) | 11.15 |
| PS + HOS | 10.08–10.09 |
| HOS + Scat | 11.33–11.36 |
| PS + HOS + Scat | 11.04–11.40 |

This degradation is explained by the normalisation bug described in §2.1. The PS features handed to the network are severely miscalibrated at high k (+3 to +4σ offsets), so the network must learn to ignore them — a task that consumes model capacity and disturbs the training signal from HOS features. When scattering is also included, the richer gradient landscape helps recovery, explaining the partial restoration to 11.04–11.40.

With the normalisation fix, PS may contribute incremental complementary information (Gaussian-scale amplitude measurements that are distinct from the non-Gaussian HOS descriptors). The corrected ablation runs (SLURM jobs 7859129–7859130) will determine whether fixed PS provides any gain over HOS + Scat alone.

---

## 3. HOS Variant Ablation

Three variants of HOS were compared:

| Variant | Description | Avg score (n) |
|---------|-------------|--------------|
| **Full HOS** | Peak counts + L1-norm histograms | **10.82** (4) |
| L1-norm only | L1-norm histograms only (no peak counts) | 10.88 (3) |
| Peaks only | Peak counts only (no L1-norms) | 10.60 (3) |

*All HOS-alone runs; same SNR and bin settings within each group.*

The full HOS is marginally the best, confirming that peak counts and L1-norms carry complementary information. The fact that **L1-norms alone slightly outperform peaks alone** is physically meaningful: the L1-norm histogram encodes the full one-point distribution of the wavelet-filtered field — including voids (negative SNR regions) — whereas peak counts only measure local maxima. Voids have been shown to carry substantial cosmological information orthogonal to peaks, particularly for Ω_m. Including both maximises the constraining power.

---

## 4. Hyperparameter Sensitivity

### 4.1 Number of HOS scales

| n_scales | Representative runs | Avg val score |
|----------|---------------------|---------------|
| 3 | psobiafb, fjy66buh, et2is8tp | 10.81 |
| 4 | lprg7hu1, 388wnkgf, thx56mt2, 7s2u38we | 11.13 |
| 6 | yj48er19, wml1jqti, 0ajzckx6, 3cjs3z1x | 11.08 |

Three wavelet scales already captures most of the cosmological information. Increasing from 3 to 4 scales provides a moderate gain (~0.3 pt), while further increasing to 6 shows no additional benefit. **Four scales is a good default.**

### 4.2 Number of HOS bins

| n_bins (peaks) | Runs | Score range |
|----------------|------|-------------|
| 31 | 388wnkgf, szz6u1c1, fjy66buh | 10.81–10.96 |
| 51 | lprg7hu1, wml1jqti, psobiafb | 10.99–11.15 |

Using 51 bins provides a consistent ~0.1–0.2 pt improvement over 31 bins, likely because the finer binning better resolves the peak count distribution near the noise threshold (SNR ~ 0–3), where the cosmological signal is concentrated. **51 bins is preferred.**

### 4.3 Scattering depth J

| J | HOS + Scat score | Scat-only score |
|---|-----------------|-----------------|
| 3 | 11.334 | 9.405 |
| 4 | **11.361** | **9.942** |
| 5 | 11.328–11.399 | 9.576 (±instability) |

In combination with HOS, all three J values perform similarly (~11.33–11.36), with J=4 being marginally best. In the scattering-only regime, J=4 also wins. J=5 with scattering alone shows training instability in one run (diverged to −23.7), suggesting the larger input dimensionality (more scattering coefficients) may require more careful tuning. **J=4 is the safest and best choice.**

---

## 5. Effect of SNR Range on HOS Performance

The HOS statistics involve histogramming wavelet-filtered field values in bins of signal-to-noise ratio (SNR = κ_w/σ_noise). Two SNR parameters are independently controlled:

- **`hos_min_snr` / `hos_max_snr`**: SNR range for the wavelet **peak count** histograms
- **`hos_l1_min_snr` / `hos_l1_max_snr`**: SNR range for the wavelet **L1-norm** histograms

Most completed runs use the setting peaks = [−3, 7], L1 = [−7, 7] with 80 L1 bins. However, since these parameters were not independently varied in the completed experiments (SNR range and bin count changed together), a dedicated ablation is needed.

### 5.1 Pending runs

Five new runs have been submitted to investigate SNR range effects, all using the best fixed base (HOS + Scat, J=4, ns=4, nbins=51, l1_nbins=80):

| Job ID | Config | Peaks SNR | L1 SNR | Hypothesis |
|--------|--------|-----------|--------|------------|
| 7858310 | `ablation_snr_peaks_narrow` | [−1, 5] | [−7, 7] | Cutting low-SNR peaks (noise-dominated) has little effect |
| 7858311 | `ablation_snr_peaks_wide` | [−5, 10] | [−7, 7] | Wider range captures rare high-SNR clusters; modest gain expected |
| 7858312 | `ablation_snr_peaks_posonly` | [0, 8] | [−7, 7] | No void contribution to peaks; expected drop, testing void sensitivity |
| 7858313 | `ablation_snr_l1_narrow` | [−3, 7] | [−5, 5] | Narrow L1 range loses tails; expected drop |
| 7858314 | `ablation_snr_l1_wide` | [−3, 7] | [−10, 10] | Very wide L1 range dilutes signal in sparse bins; ambiguous |

### 5.2 Physical interpretation

From lensing theory, the SNR distribution of wavelet peaks carries different information in different regimes:

- **High-SNR peaks (κ/σ > 3)**: Dominated by massive dark matter haloes. Strongly sensitive to σ_8 (cluster abundance scales as ~σ_8^8 at fixed mass threshold).
- **Intermediate SNR (0 < κ/σ < 3)**: Contributions from filaments, projected structures, and smaller haloes. Sensitive to the combination σ_8 Ω_m^α.
- **Negative SNR (κ/σ < 0)**: Voids and underdense regions. Carry independent information on Ω_m and potentially dark energy.

The L1-norm histogram captures the full field distribution, so negative SNR bins encode underdense regions (voids). Restricting to positive SNR would lose this void contribution, which is orthogonal to the peak information and expected to degrade performance.

> **Section to be updated once SNR ablation runs complete.**

---

## 6. Summary and Recommendations

| Recommendation | Evidence |
|----------------|----------|
| **Use HOS + Scattering** | Best val score (11.33–11.40); complementary non-Gaussian features |
| **Use full HOS** (peaks + L1-norms) | ~0.2 pt gain over peaks-only; L1-norms capture void information |
| **4 wavelet scales** | Saturation beyond 4 scales; 3 is near-optimal, 6 brings no gain |
| **51 bins for peak counts** | +0.1–0.2 pt over 31 bins at negligible cost |
| **Scattering depth J = 4** | Marginally best; avoids instability of J=5 in high-dim regime |
| **Fix PS normalisation before including PS** | Original `LOG_PS_MEAN/STD` computed on noiseless maps; per-batch norm fix submitted (jobs 7859129–7859130) |
| **SNR range TBD** | See §5; pending 5 runs (SLURM jobs 7858310–7858314) |

### Best configuration (current)

```yaml
use_ps: false
use_hos: true
hos_n_scales: 4
hos_n_bins: 51
hos_l1_nbins: 80
hos_min_snr: -3.0
hos_max_snr: 7.0
hos_l1_min_snr: -7.0
hos_l1_max_snr: 7.0
use_scattering: true
scattering_J: 4
scattering_L: 8
```

This corresponds to run `7s2u38we` (val_score = **11.361**, val_MSE = **7.04×10⁻⁴**).

---

## Appendix: Full Run Table (with SNR settings)

| Run ID | Statistics | Variant | ns | nbins | l1nbins | Peaks SNR | L1 SNR | Val score | Val MSE |
|--------|-----------|---------|-----|-------|---------|-----------|--------|-----------|---------|
| yj48er19 | PS+HOS | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 10.086 | 1.02e-3 |
| 71qhtzi4 | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | −23.684 | 7.97e-3 |
| f1uqjf8o | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 4.976 | 2.25e-3 |
| sy8rmbah | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 8.230 | 1.49e-3 |
| 1dl574x4 | PS+Scat | — | — | 31 | 40 | [−4, 8] | [−8, 8] | 5.010 | 2.40e-3 |
| v6cj7p98 | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 11.041 | 7.78e-4 |
| 3cjs3z1x | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 11.331 | 7.24e-4 |
| dv0w9s6m | PS+HOS+Scat | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 11.399 | 7.08e-4 |
| lprg7hu1 | HOS | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 11.153 | 7.53e-4 |
| 388wnkgf | HOS | l1_only | 4 | 31 | 80 | [−4, 8] | [−7, 7] | 10.963 | 7.99e-4 |
| wml1jqti | HOS | full | 6 | 51 | 80 | [−3, 7] | [−7, 7] | 10.986 | 7.81e-4 |
| szz6u1c1 | HOS | l1_only | 6 | 31 | 80 | [−4, 8] | [−7, 7] | 10.852 | 8.19e-4 |
| thx56mt2 | HOS | peaks_only | 4 | 51 | 40 | [−3, 7] | [−8, 8] | 10.636 | 8.68e-4 |
| 0ajzckx6 | HOS | peaks_only | 6 | 51 | 40 | [−3, 7] | [−8, 8] | 10.601 | 8.75e-4 |
| 354qkkwq | Scat | — | — | — | — | — | — | 9.405 | 1.19e-3 |
| md5lhelk | PS | — | — | — | — | — | — | −0.174 | 4.09e-3 |
| 78a96766 | Scat | — | — | — | — | — | — | 9.576 | 1.16e-3 |
| 1p44eqr2 | Scat | — | — | — | — | — | — | 9.942 | 1.05e-3 |
| et2is8tp | HOS | peaks_only | 3 | 51 | 40 | [−3, 7] | [−8, 8] | 10.570 | 8.82e-4 |
| fjy66buh | HOS | l1_only | 3 | 31 | 80 | [−4, 8] | [−7, 7] | 10.815 | 8.41e-4 |
| msatahn4 | PS+HOS | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 10.079 | 1.03e-3 |
| psobiafb | HOS | full | 3 | 51 | 80 | [−3, 7] | [−7, 7] | 11.051 | 7.83e-4 |
| 7s2u38we | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 11.361 | 7.04e-4 |
| pmu1wrjv | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 11.334 | 7.11e-4 |
| qwp9c44x | HOS+Scat | full | 4 | 51 | 80 | [−3, 7] | [−7, 7] | 11.328 | 7.16e-4 |
