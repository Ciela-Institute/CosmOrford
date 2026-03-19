"""
Power spectrum and higher-order summary statistics utilities for batched image data.
"""

import torch
import numpy as np

LOG_PS_MEAN = torch.tensor([ -8.676514 ,  -8.953475 ,  -9.1855755,  -9.435738 ,  -9.680172 ,
         -9.947989 , -10.258803 , -10.614475 , -10.918096 , -11.084858], dtype=torch.float32)
LOG_PS_STD = torch.tensor([0.2304908 , 0.22719958, 0.22156876, 0.23032227, 0.2407397 ,
        0.26038605, 0.2909596 , 0.3229235 , 0.35014188, 0.3699048], dtype=torch.float32)


def power_spectrum_batch(x, pixsize=2. / 60 / 180 * np.pi, kedge=np.logspace(2, 4, 11), normalize=True):
    """
    Compute the azimuthally averaged 2D power spectrum of batched real-valued 2D fields.

    Parameters:
    -----------
    x : torch.Tensor
        Input real-space maps with shape (batch, ny, nx).
        Each slice x[i] represents a 2D field (e.g., an image or simulated field).

    pixsize : float
        Physical size of each pixel in the map (e.g., arcmin, Mpc, etc.).
        Units should be consistent with the units used for `kedge`.

    kedge : torch.Tensor or array-like
        Bin edges in wavenumber space (k), used to bin the power spectrum.
        Should be monotonically increasing and cover the k-range of interest.
        Shape: (n_edges,)

    normalize : bool
        If True, normalize the power spectrum for ingestion by ML models.

    Returns:
    --------
    power_k : torch.Tensor
        The average wavenumber in each k bin (excluding the DC bin).
        Shape: (batch, nk) where nk = len(kedge) - 1

    power : torch.Tensor
        The binned, azimuthally averaged power spectrum corresponding to `power_k`.
        Normalized per unit area.
        Shape: (batch, nk)
    """

    # Ensure the input array is 3D: [batch, ny, nx]
    assert x.ndim == 3, f"Expected 3D input [batch, ny, nx], got shape {x.shape}"

    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    # Convert kedge to tensor if needed
    if not isinstance(kedge, torch.Tensor):
        kedge = torch.tensor(kedge, device=device, dtype=dtype)
    else:
        kedge = kedge.to(device=device, dtype=dtype)

    # Compute the 2D FFT and power spectrum
    xk = torch.fft.rfft2(x)  # Shape: [batch, ny, nx//2 + 1]
    xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2

    # Compute the wavenumber grid
    ky = torch.fft.fftfreq(ny, d=pixsize, device=device, dtype=dtype)
    kx = torch.fft.rfftfreq(nx, d=pixsize, device=device, dtype=dtype)

    ky_grid = ky.reshape(-1, 1) ** 2
    kx_grid = kx.reshape(1, -1) ** 2
    k = torch.sqrt(ky_grid + kx_grid) * 2 * np.pi  # Shape: (ny, nx//2 + 1)

    # Bin indices
    index = torch.searchsorted(kedge, k.flatten()).reshape(ny, nx // 2 + 1)

    # Number of bins
    n_bins = len(kedge)
    nk = n_bins - 1

    # Flatten spatial dimensions for binning
    # Shape transformations: [batch, ny, nx//2+1] -> [batch, ny*(nx//2+1)]
    xk2_flat = xk2.reshape(batch_size, -1)
    k_flat = k.flatten().unsqueeze(0).expand(batch_size, -1)  # [batch, ny*(nx//2+1)]
    index_flat = index.flatten().unsqueeze(0).expand(batch_size, -1)  # [batch, ny*(nx//2+1)]

    # Initialize accumulators
    power = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    power_k = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)
    Nmode = torch.zeros(batch_size, n_bins, device=device, dtype=dtype)

    # Scatter add to accumulate into bins
    for b in range(batch_size):
        power[b].index_add_(0, index_flat[b], xk2_flat[b])
        power_k[b].index_add_(0, index_flat[b], k_flat[b])
        Nmode[b].index_add_(0, index_flat[b], torch.ones_like(xk2_flat[b]))

    # Add mirror contributions
    if nx % 2 == 0:
        mirror_slice = slice(1, -1)
    else:
        mirror_slice = slice(1, None)

    xk2_mirror = xk2[:, :, mirror_slice].reshape(batch_size, -1)
    k_mirror = k[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)
    index_mirror = index[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)

    for b in range(batch_size):
        power[b].index_add_(0, index_mirror[b], xk2_mirror[b])
        power_k[b].index_add_(0, index_mirror[b], k_mirror[b])
        Nmode[b].index_add_(0, index_mirror[b], torch.ones_like(xk2_mirror[b]))

    # Average
    select = Nmode > 0
    power[select] = power[select] / Nmode[select]
    power_k[select] = power_k[select] / Nmode[select]
    power_k = power_k[:, 1:nk+1]  # Exclude DC bin

    # Normalize by map area
    power *= pixsize ** 2 / ny / nx
    power = power[:, 1:nk+1]# Exclude DC bin

    if normalize:
        # Normalize per-batch (consistent with HOS/scattering normalization).
        # LOG_PS_MEAN/STD were computed on noiseless maps; using them on noisy
        # maps causes +1 to +4 sigma bias at high-k (noise-dominated) bins.
        log_power = torch.log10(power + 1e-30)
        mean = log_power.mean(dim=0, keepdim=True)
        std = log_power.std(dim=0, keepdim=True) + 1e-8
        log_power = (log_power - mean) / std
        power = log_power

    return power_k, power


def compute_wavelet_peaks_batch(x, noise_std, mask=None, n_scales=5,
                                pixel_arcmin=2.0, n_bins=31,
                                min_snr=-4.0, max_snr=8.0,
                                normalize=True):
    """
    Compute ONLY wavelet peak counts for batched convergence maps (no L1-norms).

    Parameters:
    -----------
    x : torch.Tensor
        Input convergence maps with shape (batch, ny, nx).
    noise_std : float or torch.Tensor
        Noise standard deviation (scalar, (ny, nx), or (batch, ny, nx)).
    mask : torch.Tensor, optional
        Survey mask with shape (ny, nx); 1=valid, 0=masked.
    n_scales : int
        Number of wavelet scales.
    pixel_arcmin : float
        Pixel size in arcminutes.
    n_bins : int
        Number of bins for the peak count histograms.
    min_snr : float
        Minimum SNR value for peak histogram bins.
    max_snr : float
        Maximum SNR value for peak histogram bins.
    normalize : bool
        If True, apply log10(1+x) then z-score standardization.

    Returns:
    --------
    features : torch.Tensor
        Shape (batch, n_scales * n_bins).
    """
    try:
        from wl_stats_torch import WLStatistics
    except ImportError:
        raise ImportError(
            "wl_stats_torch package is required for higher-order statistics. "
            "Please install it to use this functionality."
        )

    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    stats_computer = WLStatistics(
        n_scales=n_scales,
        device=device,
        pixel_arcmin=pixel_arcmin,
        dtype=torch.float32,
    )

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        mask = mask.float()

    if isinstance(noise_std, (int, float)):
        sigma = float(noise_std)
    elif torch.is_tensor(noise_std):
        sigma = noise_std.float()
    else:
        sigma = float(noise_std)

    stats_computer.compute_wavelet_transform(x.float(), sigma, mask=mask)

    _, peaks_list = stats_computer.compute_wavelet_peak_counts(
        n_bins=n_bins,
        mask=mask,
        min_snr=min_snr,
        max_snr=max_snr,
        clamp_overflow=False,
    )

    wavelet_peaks = torch.stack(peaks_list)  # (n_scales, B, n_bins)
    all_features = wavelet_peaks.permute(1, 0, 2).flatten(1)  # (B, n_scales*n_bins)

    if normalize:
        all_features = torch.log10(all_features + 1.0)
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True) + 1e-8
        all_features = (all_features - mean) / std

    return all_features.to(dtype=dtype)


def compute_wavelet_l1_norms_batch(x, noise_std, mask=None, n_scales=5,
                                   pixel_arcmin=2.0, l1_nbins=40,
                                   l1_min_snr=-8.0, l1_max_snr=8.0,
                                   normalize=True):
    """
    Compute ONLY wavelet L1-norms for batched convergence maps (no peak counts).

    Parameters:
    -----------
    x : torch.Tensor
        Input convergence maps with shape (batch, ny, nx).
    noise_std : float or torch.Tensor
        Noise standard deviation (scalar, (ny, nx), or (batch, ny, nx)).
    mask : torch.Tensor, optional
        Survey mask with shape (ny, nx); 1=valid, 0=masked.
    n_scales : int
        Number of wavelet scales.
    pixel_arcmin : float
        Pixel size in arcminutes.
    l1_nbins : int
        Number of bins for the L1-norm histograms.
    l1_min_snr : float
        Minimum SNR value for L1-norm bins.
    l1_max_snr : float
        Maximum SNR value for L1-norm bins.
    normalize : bool
        If True, apply log10(1+x) then z-score standardization.

    Returns:
    --------
    features : torch.Tensor
        Shape (batch, n_scales * l1_nbins).
    """
    try:
        from wl_stats_torch import WLStatistics
    except ImportError:
        raise ImportError(
            "wl_stats_torch package is required for higher-order statistics. "
            "Please install it to use this functionality."
        )

    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    stats_computer = WLStatistics(
        n_scales=n_scales,
        device=device,
        pixel_arcmin=pixel_arcmin,
        dtype=torch.float32,
    )

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        mask = mask.float()

    if isinstance(noise_std, (int, float)):
        sigma = float(noise_std)
    elif torch.is_tensor(noise_std):
        sigma = noise_std.float()
    else:
        sigma = float(noise_std)

    stats_computer.compute_wavelet_transform(x.float(), sigma, mask=mask)

    l1_bins_list, l1_norms_list = stats_computer.compute_wavelet_l1_norms(
        n_bins=l1_nbins,
        mask=mask,
        min_snr=l1_min_snr,
        max_snr=l1_max_snr,
        clamp_overflow=False,
    )

    wavelet_l1 = torch.stack(l1_norms_list)  # (n_scales, B, l1_nbins)
    all_features = wavelet_l1.permute(1, 0, 2).flatten(1)  # (B, n_scales*l1_nbins)

    if normalize:
        all_features = torch.log10(all_features + 1.0)
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True) + 1e-8
        all_features = (all_features - mean) / std

    return all_features.to(dtype=dtype)


def compute_higher_order_statistics_batch(x, noise_std, mask=None, n_scales=5,
                                          pixel_arcmin=2.0, n_bins=31, l1_nbins=40,
                                          min_snr=-4.0, max_snr=8.0,
                                          l1_min_snr=-8.0, l1_max_snr=8.0,
                                          compute_mono=False, mono_smoothing_sigma=2.0,
                                          normalize=True):
    """
    Compute higher-order summary statistics for batched convergence maps.

    Includes multi-scale wavelet peak counts, wavelet L1-norms, and optionally
    mono-scale (Gaussian-smoothed) peak counts.

    Parameters:
    -----------
    x : torch.Tensor
        Input convergence maps with shape (batch, ny, nx).
    noise_std : float or torch.Tensor
        Noise standard deviation (scalar, (ny, nx), or (batch, ny, nx)).
    mask : torch.Tensor, optional
        Survey mask with shape (ny, nx); 1=valid, 0=masked.
    n_scales : int
        Number of wavelet scales.
    pixel_arcmin : float
        Pixel size in arcminutes.
    n_bins : int
        Number of bins for peak count histograms.
    l1_nbins : int
        Number of bins for L1-norm histograms.
    min_snr, max_snr : float
        SNR range for peak histogram bins.
    l1_min_snr, l1_max_snr : float
        SNR range for L1-norm bins.
    compute_mono : bool
        Whether to include mono-scale peak counts.
    mono_smoothing_sigma : float
        Gaussian smoothing sigma (pixels) for mono-scale peaks.
    normalize : bool
        If True, apply log10(1+x) then z-score standardization.

    Returns:
    --------
    features : torch.Tensor
        Shape (batch, n_features).
        n_features = n_scales*(n_bins+l1_nbins)  [+ n_bins if compute_mono]
    """
    try:
        from wl_stats_torch import WLStatistics
    except ImportError:
        raise ImportError(
            "wl_stats_torch package is required for higher-order statistics. "
            "Please install it to use this functionality."
        )

    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    stats_computer = WLStatistics(
        n_scales=n_scales,
        device=device,
        pixel_arcmin=pixel_arcmin,
        dtype=torch.float32,
    )

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(device=device, dtype=torch.float32)
        mask = mask.float()

    if isinstance(noise_std, (int, float)):
        sigma = float(noise_std)
    elif torch.is_tensor(noise_std):
        sigma = noise_std.float()
    else:
        sigma = float(noise_std)

    results = stats_computer.compute_all_statistics(
        x.float(),
        sigma,
        mask=mask,
        min_snr=min_snr,
        max_snr=max_snr,
        n_bins=n_bins,
        l1_nbins=l1_nbins,
        l1_min_snr=l1_min_snr,
        l1_max_snr=l1_max_snr,
        compute_mono=compute_mono,
        mono_smoothing_sigma=mono_smoothing_sigma,
        verbose=False,
        clamp_overflow=False,
    )

    wavelet_peaks = torch.stack(results['wavelet_peak_counts'])  # (n_scales, B, n_bins)
    wavelet_l1 = torch.stack(results['wavelet_l1_norms'])        # (n_scales, B, l1_nbins)

    if compute_mono:
        mono_peaks = results['mono_peak_counts']  # (B, n_bins)
        all_features = torch.cat([
            mono_peaks,
            wavelet_peaks.permute(1, 0, 2).flatten(1),
            wavelet_l1.permute(1, 0, 2).flatten(1),
        ], dim=1)
    else:
        all_features = torch.cat([
            wavelet_peaks.permute(1, 0, 2).flatten(1),
            wavelet_l1.permute(1, 0, 2).flatten(1),
        ], dim=1)

    if normalize:
        all_features = torch.log10(all_features + 1.0)
        mean = all_features.mean(dim=0, keepdim=True)
        std = all_features.std(dim=0, keepdim=True) + 1e-8
        all_features = (all_features - mean) / std

    return all_features.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Wavelet Scattering Transform
# ---------------------------------------------------------------------------

_scattering_cache: dict = {}


def _get_scattering_obj(J: int, L: int, H: int, W: int, device: torch.device):
    """Return a cached kymatio Scattering2D instance."""
    key = (J, L, H, W, str(device))
    if key not in _scattering_cache:
        from kymatio.torch import Scattering2D
        S = Scattering2D(J=J, L=L, shape=(H, W))
        S = S.to(device)
        _scattering_cache[key] = S
    return _scattering_cache[key]


def scattering_n_coefficients(
    J: int,
    L: int,
    feature_pooling: str = "mean",
) -> int:
    """Number of scattering coefficients for orders 0, 1, and 2.

    K = 1 (order 0) + J*L (order 1) + L^2 * J*(J-1)/2 (order 2)
    """
    base = 1 + L * J + L * L * J * (J - 1) // 2
    if feature_pooling == "mean":
        return base
    if feature_pooling == "mean_std":
        return 2 * base
    raise ValueError(
        f"Invalid feature_pooling='{feature_pooling}'. Expected one of ['mean', 'mean_std']."
    )


def scattering_order_slices(J: int, L: int) -> dict:
    """Return coefficient index slices for scattering orders 0/1/2."""
    n0 = 1
    n1 = J * L
    n2 = L * L * J * (J - 1) // 2
    return {
        "order0": slice(0, n0),
        "order1": slice(n0, n0 + n1),
        "order2": slice(n0 + n1, n0 + n1 + n2),
    }


def compute_scattering_batch(
    x,
    J=5,
    L=8,
    normalize=True,
    normalization: str = "log1p_zscore",
    mask: torch.Tensor = None,
    mask_threshold: float = 0.2,
    mask_pooling: str = "soft",
    feature_pooling: str = "mean",
):
    """
    Compute 2D wavelet scattering transform coefficients for batched maps.

    Uses the kymatio library (PyTorch frontend). The scattering object is cached
    so wavelet filters are only built once per (J, L, H, W, device) combination.

    Parameters
    ----------
    x : torch.Tensor
        Input maps with shape (batch, ny, nx).
    J : int
        Maximum wavelet scale (must satisfy 2**J <= min(ny, nx)).
    L : int
        Number of angular orientations.
    normalize : bool
        Backward-compatible flag. If True, applies the normalization selected by
        ``normalization``. If False, returns raw averaged scattering coefficients.
    normalization : str
        One of:
          - "log1p_zscore": log1p transform then per-batch z-score (default).
          - "zscore": per-batch z-score without log1p.
          - "none": no feature normalization.
    mask : torch.Tensor, optional
        Survey mask with shape (ny, nx) or (1, ny, nx). If provided, scattering
        coefficients are pooled with mask-aware weighting over the downsampled
        scattering grid instead of uniform spatial averaging.
    mask_threshold : float
        Threshold applied to the downsampled mask when ``mask_pooling='hard'``.
    mask_pooling : str
        One of:
          - "soft": weighted spatial averaging using fractional mask coverage.
          - "hard": binary include/exclude with ``mask_threshold``.
    feature_pooling : str
        One of:
          - "mean": use only spatial mean of each scattering coefficient.
          - "mean_std": concatenate spatial mean and spatial std per coefficient.

    Returns
    -------
    features : torch.Tensor
        Shape (batch, K) for ``feature_pooling='mean'`` and (batch, 2K) for
        ``feature_pooling='mean_std'``, where
        K = 1 + L*J + L**2 * J*(J-1)//2.
    """
    assert x.ndim == 3, f"Expected 3D input (batch, ny, nx), got shape {x.shape}"
    batch_size, ny, nx = x.shape
    device = x.device
    dtype = x.dtype

    assert 2 ** J <= min(ny, nx), (
        f"J={J} too large for map size ({ny}, {nx}): need 2**J={2**J} <= {min(ny, nx)}"
    )

    x_in = x.unsqueeze(1).float()  # (B, 1, ny, nx)
    scattering = _get_scattering_obj(J, L, ny, nx, device)
    Sx = scattering(x_in)          # (B, 1, K, h, w)
    expected_k = scattering_n_coefficients(J, L, feature_pooling="mean")
    actual_k = Sx.shape[2]
    if actual_k != expected_k:
        raise RuntimeError(
            f"Unexpected scattering coefficient count: expected {expected_k}, got {actual_k} "
            f"for J={J}, L={L}."
        )
    valid_feature_pooling = {"mean", "mean_std"}
    if feature_pooling not in valid_feature_pooling:
        raise ValueError(
            f"Invalid feature_pooling='{feature_pooling}'. Expected one of {sorted(valid_feature_pooling)}."
        )

    if mask is None:
        mean = Sx.mean(dim=(-2, -1)).squeeze(1)  # (B, K)
        if feature_pooling == "mean_std":
            std = Sx.std(dim=(-2, -1)).squeeze(1)
            features = torch.cat([mean, std], dim=1)
        else:
            features = mean
    else:
        valid_pooling = {"soft", "hard"}
        if mask_pooling not in valid_pooling:
            raise ValueError(
                f"Invalid mask_pooling='{mask_pooling}'. Expected one of {sorted(valid_pooling)}."
            )
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"mask must have shape (ny, nx) or (1, ny, nx), got {mask.shape}")
        mask_f = mask.to(device=device, dtype=Sx.dtype).unsqueeze(1)  # (1, 1, ny, nx)
        # Match scattering spatial grid exactly via adaptive pooling.
        mask_low = torch.nn.functional.adaptive_avg_pool2d(mask_f, Sx.shape[-2:])
        if mask_pooling == "hard":
            mask_low = (mask_low >= mask_threshold).to(Sx.dtype)  # (1,1,h,w)
        else:
            mask_low = mask_low.clamp_(0.0, 1.0)
        if mask_low.sum() <= 0:
            raise ValueError(
                "Mask has no valid support on scattering grid after downsampling. "
                f"Input mask shape={tuple(mask.shape)}, scattering grid={tuple(Sx.shape[-2:])}."
            )
        mask_low = mask_low.unsqueeze(2)  # (1,1,1,h,w), broadcast over batch/K
        denom = mask_low.sum(dim=(-2, -1)).clamp_min(1e-6)
        mean = ((Sx * mask_low).sum(dim=(-2, -1)) / denom).squeeze(1)  # (B, K)
        if feature_pooling == "mean_std":
            sq = (Sx ** 2 * mask_low).sum(dim=(-2, -1)) / denom
            var = (sq.squeeze(1) - mean ** 2).clamp_min(0.0)
            std = torch.sqrt(var + 1e-8)
            features = torch.cat([mean, std], dim=1)
        else:
            features = mean

    valid_norms = {"log1p_zscore", "zscore", "none"}
    if normalization not in valid_norms:
        raise ValueError(
            f"Invalid normalization='{normalization}'. Expected one of {sorted(valid_norms)}."
        )

    if normalize and normalization != "none":
        if normalization == "log1p_zscore":
            features = torch.log1p(features)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-8
        features = (features - mean) / std

    if not torch.isfinite(features).all():
        raise RuntimeError("Non-finite scattering features detected after pooling/normalization.")

    return features.to(dtype=dtype)
