"""Offline, deterministic measurement engine for decoded monochrome frames.

All inputs are raw, already-unpacked pixel samples in digital numbers (DN).
Packed Mono10/Mono12 buffers, RGB frames, floating-point or display-normalized
images are rejected. No conversion, clipping, normalization or rounding is
applied to the input data.

Storage dtype per bit depth (strict):
    8-bit          -> uint8
    10/12/14/16-bit -> uint16 (values must stay within [0, 2**bit_depth - 1];
                       a 10-bit frame is NOT assumed to fill the uint16 range)

Units used in returned dictionaries:
    DN          digital number (raw pixel value)
    fraction    dimensionless ratio in [0, 1]
    pixels      pixel count
"""

import math

import numpy as np

SUPPORTED_BIT_DEPTHS = (8, 10, 12, 14, 16)


def _expected_dtype(bit_depth: int) -> np.dtype:
    return np.dtype(np.uint8) if bit_depth == 8 else np.dtype(np.uint16)


def _validate_bit_depth(bit_depth: int) -> None:
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int):
        raise TypeError(f"bit_depth must be an int, got {type(bit_depth).__name__}")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(
            f"Unsupported bit_depth {bit_depth}; expected one of {SUPPORTED_BIT_DEPTHS}"
        )


def _validate_frame(frame: np.ndarray, bit_depth: int, name: str = "frame") -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame).__name__}")
    if frame.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D monochrome array (height, width); got shape "
            f"{frame.shape}. RGB/multi-channel frames are not supported."
        )
    if frame.size == 0:
        raise ValueError(f"{name} must not be empty; got shape {frame.shape}")
    if frame.dtype.kind != "u":
        raise TypeError(
            f"{name} must contain unsigned integer pixel samples; got dtype {frame.dtype}"
        )
    expected = _expected_dtype(bit_depth)
    if frame.dtype != expected:
        raise TypeError(
            f"{name} dtype {frame.dtype} does not match {bit_depth}-bit data; "
            f"expected {expected}"
        )
    max_dn = (1 << bit_depth) - 1
    observed_max = int(frame.max())
    if observed_max > max_dn:
        raise ValueError(
            f"{name} contains value {observed_max} DN exceeding the {bit_depth}-bit "
            f"maximum of {max_dn} DN (packed or mislabelled data?)"
        )


def analyze_frame(frame: np.ndarray, bit_depth: int, bins: int = 64) -> dict:
    """Compute single-frame intensity statistics.

    Returned keys and units:
        width, height                  pixels
        bit_depth                      bits
        full_scale_dn                  DN, 2**bit_depth - 1
        min_dn, max_dn                 DN
        mean_dn                        DN
        mean_fraction_of_full_scale    fraction, mean_dn / full_scale_dn
        spatial_std_dn                 DN, population std (ddof=0) over all
                                       pixels of this single frame. This is
                                       spatial variation (scene + fixed
                                       pattern + noise), NOT temporal noise.
        saturation_fraction            fraction of pixels == full_scale_dn
        zero_fraction                  fraction of pixels == 0
        percentiles_dn                 {"p1", "p50", "p99"} in DN, numpy
                                       linear interpolation
        histogram                      {"bins", "range_dn", "bin_edges_dn",
                                       "counts"}; range is [0, 2**bit_depth]
                                       split into `bins` equal-width bins.
                                       Every bin is half-open [lo, hi) except
                                       the last, which is closed. counts are
                                       pixel counts summing to width*height.
    """
    _validate_bit_depth(bit_depth)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive int, got {bins!r}")
    _validate_frame(frame, bit_depth)

    full_scale = (1 << bit_depth) - 1
    height, width = frame.shape
    n_pixels = frame.size
    mean_dn = float(frame.mean(dtype=np.float64))
    p1, p50, p99 = np.percentile(frame, [1, 50, 99])
    hist_range = (0, 1 << bit_depth)
    counts, edges = np.histogram(frame, bins=bins, range=hist_range)

    return {
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "full_scale_dn": int(full_scale),
        "min_dn": int(frame.min()),
        "max_dn": int(frame.max()),
        "mean_dn": mean_dn,
        "mean_fraction_of_full_scale": mean_dn / full_scale,
        "spatial_std_dn": float(frame.std(dtype=np.float64)),
        "saturation_fraction": int(np.count_nonzero(frame == full_scale)) / n_pixels,
        "zero_fraction": int(np.count_nonzero(frame == 0)) / n_pixels,
        "percentiles_dn": {"p1": float(p1), "p50": float(p50), "p99": float(p99)},
        "histogram": {
            "bins": int(bins),
            "range_dn": [int(hist_range[0]), int(hist_range[1])],
            "bin_edges_dn": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


def analyze_temporal_pair(
    frame_a: np.ndarray, frame_b: np.ndarray, bit_depth: int
) -> dict:
    """Compare two frames of the same scene taken at different times.

    The difference d = frame_a - frame_b is computed in signed int64, so no
    unsigned wraparound occurs and the inputs are not modified.

    Returned keys and units:
        mean_difference_dn        DN, mean(d); positive means frame_a brighter
                                  (brightness drift between frames)
        difference_spatial_std_dn DN, population std (ddof=0) of d about its
                                  mean
        temporal_noise_proxy_dn   DN, sqrt(Var(d) / 2) with ddof=0 after
                                  removing mean(d)

    The temporal noise proxy estimates per-frame random temporal fluctuation
    only if illumination is stable, the scene is stationary, and the noise in
    the two frames is independent and identically distributed. It includes
    shot noise and all other temporal noise sources and is NOT automatically
    equivalent to sensor read noise.
    """
    _validate_bit_depth(bit_depth)
    _validate_frame(frame_a, bit_depth, "frame_a")
    _validate_frame(frame_b, bit_depth, "frame_b")
    if frame_a.shape != frame_b.shape:
        raise ValueError(
            f"Frame shapes differ: frame_a {frame_a.shape} vs frame_b {frame_b.shape}"
        )
    if frame_a.dtype != frame_b.dtype:
        raise TypeError(
            f"Frame dtypes differ: frame_a {frame_a.dtype} vs frame_b {frame_b.dtype}"
        )

    diff = frame_a.astype(np.int64) - frame_b.astype(np.int64)
    mean_diff = float(diff.mean(dtype=np.float64))
    var_diff = float(np.mean((diff - mean_diff) ** 2))

    return {
        "width": int(frame_a.shape[1]),
        "height": int(frame_a.shape[0]),
        "bit_depth": int(bit_depth),
        "mean_difference_dn": mean_diff,
        "difference_spatial_std_dn": math.sqrt(var_diff),
        "temporal_noise_proxy_dn": math.sqrt(var_diff / 2.0),
    }
