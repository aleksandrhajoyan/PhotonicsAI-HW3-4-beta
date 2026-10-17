"""Offline flicker analysis of a timestamped series of per-frame mean brightness values.

Pure, deterministic bookkeeping on NumPy arrays: this module never touches a
camera, a vendor SDK, MCP, LangGraph, the network or an LLM. It estimates the
fundamental flicker frequency of a light source from frame-level samples and
reports, alongside the estimate, every reason the estimate may be unreliable.

Entry point:

    analyze_temporal_series(*, host_t_ns, mean_dn, block_ids=None,
                            device_ts=None, device_tick_hz=None,
                            reference_hz=None) -> dict

Inputs (one entry per received frame, in acquisition order):
    host_t_ns       monotonic host timestamps, nanoseconds (int or float)
    mean_dn         per-frame mean digital number (DN)
    block_ids       optional native SDK frame identifiers (integers). A gap
                    in consecutive ids is a dropped frame.
    device_ts       optional unscaled camera timestamps (ticks of unknown unit)
    device_tick_hz  optional tick frequency of device_ts, verified outside this
                    module. Without it device_ts is never converted to seconds.
    reference_hz    optional independently set stimulus frequency. It is used
                    only for the reference_validation block AFTER the estimate
                    is made; it never influences peak selection, status,
                    estimate, acquisition or timing.

Malformed calls (wrong types, non-1-D arrays, length mismatch, non-positive
device_tick_hz or reference_hz, device_tick_hz without device_ts) raise
TypeError/ValueError. Problems with the measured data itself are reported
through the returned status and warnings, never raised.

Time base:
    The HuaRay IMV_FrameInfo.timeStamp unit is not established by the vendor
    headers, so device timestamps are used for physical time ONLY when
    device_tick_hz is supplied AND the implied tick rate measured against the
    host clock, (device_ts[-1] - device_ts[0]) / host duration, agrees with it
    within DEVICE_TICK_TOLERANCE. Otherwise host timestamps are used and the
    reason is reported. Host timestamps are arrival times (USB, driver and
    Python latency included), not exposure times.

Sampling integrity:
    The spectrum assumes uniform sampling. The series is classified
    "irregular_sampling" and NO spectral estimate is produced when frames were
    dropped (block_id gaps), when any interval lies outside
    [GAP_LOW, GAP_HIGH] x median interval, or when the relative interval jitter
    std/median exceeds MAX_JITTER_REL. Nothing is interpolated or repaired.
    MAX_JITTER_REL = 0.10: a timing error of sigma seconds shifts the phase of
    a component at f by 2*pi*f*sigma; at Nyquist that is pi*0.10 = 0.31 rad
    rms, an amplitude loss of about exp(-0.31**2 / 2) = 5 %, and less at lower
    frequencies.

Spectrum and fundamental estimate (status "ok" / "ambiguous"):
    1. Remove a least-squares line (detrend), apply a Hann window and
       zero-pad to the next power of two >= ZERO_PAD_FACTOR * N.
    2. fs = (N - 1) / duration. Native resolution resolution_hz = fs / N
       (about 1 / record length). Search band [SEARCH_FLOOR_BINS *
       resolution_hz, fs / 2]: below it the Hann main lobe of the residual
       mean dominates, and a component there would have fewer than about three
       cycles in the record.
    3. Spectral quality metric snr: peak power divided by the estimated mean
       noise power per spectral point, noise_mean = median(power in search
       band) / ln 2 (for Gaussian noise each point's power is exponential, whose
       median is ln 2 times its mean). For pure noise, snr ~ Exp(1), so the
       chance that any of n_independent = (band width / resolution_hz) bins
       exceeds x is about n_independent * exp(-x). The detection threshold is
       ln(n_independent) + ln(1 / FALSE_ALARM_PROBABILITY), at least SNR_MIN.
    4. Significant peaks: local maxima above the threshold, at least
       PEAK_REL_MIN of the strongest peak's power (a 50 % square wave's 9th
       harmonic is 1/81 = 0.012), and not within PEAK_MASK_BINS of a stronger
       accepted peak. Frequencies are refined by parabolic interpolation of
       log power.
    5. Every significant peak is a candidate fundamental, the lowest one
       included. A candidate is supported when every significant peak is
       explained as harmonic m = 1..HARMONIC_LIMIT of it, directly or folded
       about multiples of fs (aliased), within HARMONIC_TOL_BINS *
       resolution_hz, AND every lower peak (which can only be an aliased
       harmonic of it, since frame sampling can fold e.g. |fs - 3f| below f)
       is weaker than it. The lowest peak needs no strength check.
    6. f0_hz is reported only if EXACTLY ONE candidate is supported; a
       supported candidate that is not the lowest peak is reported with
       LOWER_PEAK_IS_ALIASED_HARMONIC. Two or more supported candidates
       (COMPETING_FUNDAMENTALS: e.g. a weak alias that by chance also
       explains f as one of its own harmonics) or none (missing fundamental,
       two unrelated sources, an "alias" stronger than its fundamental) give
       status "ambiguous" with f0_hz None. The candidates are measured peaks,
       so f0_hz never exceeds Nyquist.
    tolerance_hz = resolution_hz: two components closer than 1 / T cannot be
    resolved, so no claim finer than one native bin is made.

Aliasing: frame sampling cannot tell f from |f - k * fs|. Every estimate
assumes the true frequency is below Nyquist and says so in its warnings. The
original frequency of an aliased signal is never inferred from its spectrum.
If reference_hz >= Nyquist, reference_validation reports it as not
recoverable and gives only the alias position that reference would produce.

Result (JSON-compatible dict; future lumen.flicker.v1 mapping noted):
    status                 one of STATUSES                       -> status
    warnings               [{"code", "message"}]                 -> warnings
    acquisition            sample_kind "frames" (-> acquisition.sample_kind),
                           n_frames, duration_s, fps (-> acquisition.fps),
                           fps_host, fps_device, nyquist_hz, timestamp_source.
                           exposure_us is deliberately absent: the acquisition
                           layer supplies it.
    timing                 interval statistics, jitter, dropped frames,
                           device timestamp scale check
    estimate               f0_hz (-> estimate.f0_hz), resolution_hz
                           (-> estimate.resolution_hz), snr (-> estimate.snr),
                           tolerance_hz, snr_db, snr_threshold, peaks, ...
    reference_validation   None, or the post-hoc comparison with reference_hz
"""

import math

import numpy as np

STATUS_OK = "ok"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_NO_FLICKER = "no_flicker_detected"
STATUS_INSUFFICIENT = "insufficient_data"
STATUS_IRREGULAR = "irregular_sampling"
STATUS_INVALID = "invalid_input"
STATUSES = (STATUS_OK, STATUS_AMBIGUOUS, STATUS_NO_FLICKER, STATUS_INSUFFICIENT,
            STATUS_IRREGULAR, STATUS_INVALID)

SOURCE_HOST = "host_monotonic"
SOURCE_DEVICE = "device_verified"

MIN_FRAMES = 32
MAX_JITTER_REL = 0.10
GAP_LOW = 0.5
GAP_HIGH = 1.5
DEVICE_TICK_TOLERANCE = 0.02
ZERO_PAD_FACTOR = 8
SEARCH_FLOOR_BINS = 3.0
FALSE_ALARM_PROBABILITY = 1e-3
SNR_MIN = 10.0
PEAK_REL_MIN = 1e-2
PEAK_MASK_BINS = 2.5
HARMONIC_LIMIT = 10
HARMONIC_TOL_BINS = 1.5
NEAR_NYQUIST_FRACTION = 0.8
MAX_REPORTED_PEAKS = 10


def analyze_temporal_series(*, host_t_ns, mean_dn, block_ids=None, device_ts=None,
                            device_tick_hz=None, reference_hz=None):
    """Estimate the flicker fundamental of a per-frame brightness series. See the module docstring."""
    host = _as_1d("host_t_ns", host_t_ns, "iuf")
    signal = _as_1d("mean_dn", mean_dn, "iuf")
    n = len(host)
    if len(signal) != n:
        raise ValueError(f"mean_dn has {len(signal)} entries, host_t_ns has {n}")
    blocks = None if block_ids is None else _as_1d("block_ids", block_ids, "iu")
    ticks = None if device_ts is None else _as_1d("device_ts", device_ts, "iuf")
    for name, array in (("block_ids", blocks), ("device_ts", ticks)):
        if array is not None and len(array) != n:
            raise ValueError(f"{name} has {len(array)} entries, host_t_ns has {n}")
    tick_hz = _optional_positive("device_tick_hz", device_tick_hz)
    ref = _optional_positive("reference_hz", reference_hz)
    if tick_hz is not None and ticks is None:
        raise ValueError("device_tick_hz requires device_ts")

    result = _skeleton(n)
    warnings = result["warnings"]

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    def finish(status):
        result["status"] = status
        result["reference_validation"] = _reference_validation(ref, result, warn)
        return result

    # -- validity ----------------------------------------------------------
    invalid = False
    for name, array in (("host_t_ns", host), ("mean_dn", signal), ("device_ts", ticks)):
        if array is not None and array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            warn("NONFINITE_VALUES", f"{name} contains NaN or infinite values")
            invalid = True
    if invalid:
        return finish(STATUS_INVALID)
    for name, array in (("host_t_ns", host), ("device_ts", ticks), ("block_ids", blocks)):
        if array is None or n < 2:
            continue
        if np.any(array[1:] == array[:-1]):
            warn("DUPLICATE_TIMESTAMPS" if name != "block_ids" else "DUPLICATE_BLOCK_IDS",
                 f"{name} contains repeated consecutive values")
            invalid = True
        if np.any(array[1:] < array[:-1]):
            warn("NONMONOTONIC_TIMESTAMPS" if name != "block_ids" else "NONMONOTONIC_BLOCK_IDS",
                 f"{name} decreases at least once (reordered data or a counter reset)")
            invalid = True
    if invalid:
        return finish(STATUS_INVALID)
    if n < MIN_FRAMES:
        warn("TOO_FEW_FRAMES", f"{n} frames received; at least {MIN_FRAMES} are required")
        return finish(STATUS_INSUFFICIENT)

    # -- time base ---------------------------------------------------------
    host_s = _offsets(host) * 1e-9
    host_duration = float(host_s[-1])
    acquisition, timing = result["acquisition"], result["timing"]
    acquisition["fps_host"] = (n - 1) / host_duration
    time_s, source = host_s, SOURCE_HOST
    device = timing["device_timestamps"]
    if ticks is not None:
        tick_offsets = _offsets(ticks)
        implied = float(tick_offsets[-1]) / host_duration
        device.update(provided=True, declared_tick_hz=tick_hz, implied_tick_hz=implied)
        if tick_hz is None:
            warn("DEVICE_TICK_UNVERIFIED",
                 "device_ts given without a verified device_tick_hz; its unit is unknown, so it is "
                 "not converted to seconds and host timestamps are used")
        else:
            relative = abs(implied / tick_hz - 1.0)
            device["relative_difference"] = relative
            if relative <= DEVICE_TICK_TOLERANCE:
                device["scale_consistent"] = True
                device["used"] = True
                time_s, source = tick_offsets / tick_hz, SOURCE_DEVICE
                acquisition["fps_device"] = (n - 1) / float(time_s[-1])
            else:
                device["scale_consistent"] = False
                warn("DEVICE_TICK_SCALE_MISMATCH",
                     f"declared device_tick_hz {tick_hz!r} disagrees with the host-implied "
                     f"{implied:.6g} Hz by {relative:.1%} (> {DEVICE_TICK_TOLERANCE:.0%}); device "
                     "timestamps are not used")
    if source == SOURCE_HOST:
        warn("HOST_TIMESTAMPS",
             "intervals and fps are host arrival times, including USB, driver and Python latency")

    duration = float(time_s[-1])
    fps = (n - 1) / duration
    nyquist = fps / 2.0
    acquisition.update(duration_s=duration, fps=fps, nyquist_hz=nyquist, timestamp_source=source)
    timing["timestamp_source"] = source

    # -- sampling integrity ------------------------------------------------
    intervals = np.diff(time_s)
    median = float(np.median(intervals))
    jitter_rel = float(np.std(intervals)) / median
    gaps = int(np.count_nonzero((intervals > GAP_HIGH * median) | (intervals < GAP_LOW * median)))
    timing.update(
        n_intervals=int(n - 1),
        median_interval_s=median,
        mean_interval_s=float(np.mean(intervals)),
        min_interval_s=float(np.min(intervals)),
        max_interval_s=float(np.max(intervals)),
        interval_std_s=float(np.std(intervals)),
        jitter_rel=jitter_rel,
        interval_gap_count=gaps,
    )
    irregular = False
    if blocks is None:
        warn("BLOCK_IDS_UNAVAILABLE",
             "no native block ids; dropped frames are detected only through interval gaps")
    else:
        dropped = int(np.sum(np.diff(_offsets(blocks)) - 1))
        timing["block_ids_available"] = True
        timing["dropped_frames"] = dropped
        if dropped:
            warn("DROPPED_FRAMES", f"{dropped} frame(s) missing according to native block ids")
            irregular = True
    if gaps:
        warn("INTERVAL_GAPS",
             f"{gaps} interval(s) outside [{GAP_LOW}, {GAP_HIGH}] x the median interval")
        irregular = True
    if jitter_rel > MAX_JITTER_REL:
        warn("EXCESSIVE_JITTER",
             f"interval jitter std/median {jitter_rel:.1%} exceeds {MAX_JITTER_REL:.0%}")
        irregular = True
    if irregular:
        warn("SPECTRUM_NOT_COMPUTED",
             "samples are not uniformly spaced; no uniform-sampling spectrum is computed and "
             "nothing is interpolated")
        return finish(STATUS_IRREGULAR)

    # -- spectrum ----------------------------------------------------------
    estimate = result["estimate"]
    resolution = fps / n
    floor_hz = SEARCH_FLOOR_BINS * resolution
    estimate.update(resolution_hz=resolution, tolerance_hz=resolution,
                    search_band_hz=[floor_hz, nyquist])
    if floor_hz >= nyquist:
        warn("RECORD_TOO_SHORT", "the search band above the DC main lobe is empty")
        return finish(STATUS_INSUFFICIENT)

    values = signal.astype(np.float64)
    index = np.arange(n, dtype=np.float64)
    residual = values - np.polyval(np.polyfit(index, values, 1), index)
    if float(np.max(np.abs(residual))) <= 1e-9 * max(1.0, float(np.max(np.abs(values)))):
        warn("NO_VARIATION_AFTER_DETREND", "the series is constant (or a straight line)")
        return finish(STATUS_NO_FLICKER)

    nfft = 1 << int(math.ceil(math.log2(ZERO_PAD_FACTOR * n)))
    power = np.abs(np.fft.rfft(residual * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    band = np.flatnonzero((freqs >= floor_hz) & (freqs <= nyquist))
    n_independent = max(1, int((nyquist - floor_hz) / resolution))
    threshold = max(SNR_MIN, math.log(n_independent) + math.log(1.0 / FALSE_ALARM_PROBABILITY))
    noise_mean = float(np.median(power[band])) / math.log(2.0)
    noise_mean = max(noise_mean, float(np.max(power[band])) * 1e-30, np.finfo(float).tiny)
    estimate.update(snr_threshold=threshold, method=(
        "linear detrend, Hann window, zero-padded FFT; lowest significant peak checked "
        "for harmonic consistency"))

    maxima = [i for i in band if power[i] >= power[i - 1] and (i + 1 >= len(power) or power[i] > power[i + 1])]
    maxima.sort(key=lambda i: power[i], reverse=True)
    if not maxima:
        warn("NO_SIGNIFICANT_PEAK", "no spectral peak in the search band")
        return finish(STATUS_NO_FLICKER)
    strongest_snr = float(power[maxima[0]]) / noise_mean
    estimate.update(snr=strongest_snr, snr_db=10.0 * math.log10(strongest_snr))

    peaks = []
    for i in maxima:
        snr = float(power[i]) / noise_mean
        if snr < threshold or power[i] < PEAK_REL_MIN * power[maxima[0]]:
            break  # sorted by power: every later maximum is weaker still
        frequency = _refine(power, freqs, i)
        if all(abs(frequency - p["frequency_hz"]) > PEAK_MASK_BINS * resolution for p in peaks):
            peaks.append({"frequency_hz": frequency, "snr": snr})
    if not peaks:
        warn("NO_SIGNIFICANT_PEAK",
             f"strongest peak snr {strongest_snr:.3g} is below the threshold {threshold:.3g}")
        return finish(STATUS_NO_FLICKER)

    strongest = peaks[0]
    estimate["strongest_peak_hz"] = strongest["frequency_hz"]
    estimate["peaks"] = sorted(peaks, key=lambda p: p["frequency_hz"])[:MAX_REPORTED_PEAKS]
    warn("ALIASING_NOT_EXCLUDED",
         f"frame sampling at {fps:.6g} fps cannot distinguish f from |f - k*fs|; the estimate "
         f"assumes the true frequency is below the {nyquist:.6g} Hz Nyquist limit")

    lowest = min(p["frequency_hz"] for p in peaks)
    tolerance = HARMONIC_TOL_BINS * resolution
    candidate = _unique_fundamental(peaks, fps, tolerance, warn)
    if candidate is None:
        estimate["candidate_f0_hz"] = lowest
        unexplained = [p["frequency_hz"] for p in peaks
                       if not _is_harmonic(p["frequency_hz"], lowest, fps, tolerance)]
        if unexplained:
            warn("HARMONIC_AMBIGUITY",
                 f"peaks at {_hz_list(unexplained)} Hz are not harmonics (direct or aliased, up to "
                 f"order {HARMONIC_LIMIT}) of the lowest peak {lowest:.6g} Hz; the fundamental "
                 "is not determined")
        return finish(STATUS_AMBIGUOUS)

    estimate["f0_hz"] = candidate
    ratio = strongest["frequency_hz"] / candidate
    order = int(round(ratio))
    if abs(strongest["frequency_hz"] - order * candidate) <= tolerance:
        estimate["harmonic_number_of_strongest"] = order
    if estimate["harmonic_number_of_strongest"] != 1:
        warn("STRONGEST_PEAK_IS_HARMONIC",
             f"the strongest peak {strongest['frequency_hz']:.6g} Hz is a harmonic of the reported "
             f"fundamental {candidate:.6g} Hz, not the fundamental itself")
    if any(abs(p["frequency_hz"] - round(p["frequency_hz"] / candidate) * candidate) > tolerance
           for p in peaks):
        warn("ALIASED_HARMONICS_PRESENT", "some peaks are explained only as aliased harmonics")
    if candidate > NEAR_NYQUIST_FRACTION * nyquist:
        warn("NEAR_NYQUIST", f"f0 {candidate:.6g} Hz is above {NEAR_NYQUIST_FRACTION:.0%} of Nyquist")
    return finish(STATUS_OK)


# -- helpers -----------------------------------------------------------------

def _as_1d(name, values, kinds):
    if values is None:
        raise TypeError(f"{name} is required")
    array = np.asarray(values)
    if array.dtype.kind not in kinds:
        raise TypeError(f"{name} must contain {'integers' if kinds == 'iu' else 'real numbers'}, "
                        f"got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _optional_positive(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _offsets(array):
    """array - array[0] as float64, subtracting in the integer domain first (no uint64 wrap, no precision loss)."""
    return (array - array[0]).astype(np.float64)


def _alias(frequency, fs):
    return abs(frequency - fs * round(frequency / fs))


def _is_harmonic(frequency, fundamental, fs, tolerance):
    return any(abs(_alias(m * fundamental, fs) - frequency) <= tolerance
               for m in range(1, HARMONIC_LIMIT + 1))


def _harmonic_order(frequency, fundamental, fs, tolerance):
    return next(m for m in range(1, HARMONIC_LIMIT + 1)
                if abs(_alias(m * fundamental, fs) - frequency) <= tolerance)


def _unique_fundamental(peaks, fs, tolerance, warn):
    """The frequency of the only supported candidate: a peak that explains every peak as a direct or aliased
    harmonic, with every lower peak weaker than it. None (with a warning saying why, when some peak explains
    every peak) if no candidate or more than one is supported."""
    consistent = [c for c in peaks
                  if all(_is_harmonic(p["frequency_hz"], c["frequency_hz"], fs, tolerance) for p in peaks)]

    def stronger_lower(c):
        return [p["frequency_hz"] for p in peaks
                if p["frequency_hz"] < c["frequency_hz"] and p["snr"] >= c["snr"]]

    supported = [c for c in consistent if not stronger_lower(c)]
    if len(supported) > 1:
        warn("COMPETING_FUNDAMENTALS",
             f"peaks at {_hz_list(sorted(c['frequency_hz'] for c in supported))} Hz each explain every peak as "
             "a direct or aliased harmonic, with every lower peak weaker; the fundamental is not determined")
        return None
    if not supported:
        for c in consistent:
            warn("ALIAS_STRONGER_THAN_CANDIDATE",
                 f"peaks at {_hz_list(stronger_lower(c))} Hz would be aliased harmonics of "
                 f"{c['frequency_hz']:.6g} Hz but are not weaker than it")
        return None
    chosen = supported[0]
    lower = [p for p in peaks if p["frequency_hz"] < chosen["frequency_hz"]]
    if not lower:
        return chosen["frequency_hz"]
    orders = ", ".join(f"{p['frequency_hz']:.6g} Hz = harmonic "
                       f"{_harmonic_order(p['frequency_hz'], chosen['frequency_hz'], fs, tolerance)}"
                       for p in lower)
    warn("LOWER_PEAK_IS_ALIASED_HARMONIC",
         f"the lowest peak is not the fundamental; {orders} of {chosen['frequency_hz']:.6g} Hz, folded about "
         f"multiples of {fs:.6g} fps, and weaker than it")
    return chosen["frequency_hz"]


def _refine(power, freqs, i):
    """Parabolic interpolation of log power around a zero-padded maximum."""
    if i <= 0 or i + 1 >= len(power) or min(power[i - 1], power[i], power[i + 1]) <= 0:
        return float(freqs[i])
    a, b, c = (math.log(float(power[j])) for j in (i - 1, i, i + 1))
    denominator = a - 2.0 * b + c
    shift = 0.0 if denominator == 0 else 0.5 * (a - c) / denominator
    return float(freqs[i] + shift * (freqs[1] - freqs[0]))


def _hz_list(values):
    return ", ".join(f"{v:.6g}" for v in values)


def _skeleton(n):
    return {
        "status": None,
        "warnings": [],
        "acquisition": {
            "sample_kind": "frames",
            "n_frames": int(n),
            "duration_s": None,
            "fps": None,
            "fps_host": None,
            "fps_device": None,
            "nyquist_hz": None,
            "timestamp_source": None,
        },
        "timing": {
            "timestamp_source": None,
            "n_intervals": None,
            "median_interval_s": None,
            "mean_interval_s": None,
            "min_interval_s": None,
            "max_interval_s": None,
            "interval_std_s": None,
            "jitter_rel": None,
            "interval_gap_count": None,
            "block_ids_available": False,
            "dropped_frames": None,
            "device_timestamps": {
                "provided": False,
                "declared_tick_hz": None,
                "implied_tick_hz": None,
                "relative_difference": None,
                "scale_consistent": None,
                "used": False,
            },
        },
        "estimate": {
            "f0_hz": None,
            "resolution_hz": None,
            "tolerance_hz": None,
            "snr": None,
            "snr_db": None,
            "snr_threshold": None,
            "strongest_peak_hz": None,
            "harmonic_number_of_strongest": None,
            "candidate_f0_hz": None,
            "peaks": [],
            "search_band_hz": None,
            "method": None,
        },
        "reference_validation": None,
    }


def _reference_validation(ref, result, warn):
    """Post-hoc comparison with an independent stimulus frequency. Never alters the estimate."""
    if ref is None:
        return None
    nyquist = result["acquisition"]["nyquist_hz"]
    estimate = result["estimate"]
    check = {
        "reference_hz": ref,
        "nyquist_hz": nyquist,
        "used_for_estimation": False,
        "recoverable": None,
        "expected_alias_hz": None,
        "abs_error_hz": None,
        "tolerance_hz": estimate["tolerance_hz"],
        "verdict": "no_estimate",
    }
    if nyquist is None:
        return check
    if ref >= nyquist:
        check["recoverable"] = False
        check["expected_alias_hz"] = _alias(ref, 2.0 * nyquist)
        check["verdict"] = "not_recoverable"
        warn("REFERENCE_AT_OR_ABOVE_NYQUIST",
             f"reference {ref:.6g} Hz is at or above the measured Nyquist {nyquist:.6g} Hz; it is not "
             "reliably recoverable with frame-level temporal sampling")
        return check
    check["recoverable"] = True
    if estimate["f0_hz"] is not None:
        error = abs(estimate["f0_hz"] - ref)
        check["abs_error_hz"] = error
        check["verdict"] = "match" if error <= estimate["tolerance_hz"] else "mismatch"
    return check
