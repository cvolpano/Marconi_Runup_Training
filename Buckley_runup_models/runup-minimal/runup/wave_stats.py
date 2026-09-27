"""
Wave statistics from a water-level time series.

Extracted from coastal_hazards_instrumentation/python/waves/spectral_functions.py
and trimmed to only the functions used by timestack2netcdf.py.

External dependencies removed:
  - statsmodels (was used only for quad_detrend; replaced with numpy polyfit)

Public API (matches the original):
  wave_stats(twl, dt, *, h, IG_divide, rho, g, nperseg) -> dict-of-dicts
"""

import numpy as np
import scipy.signal
from scipy.integrate import trapezoid
from scipy.signal import find_peaks


# ── helpers ───────────────────────────────────────────────────────────────────

def _quad_detrend(x: np.ndarray) -> np.ndarray:
    """Quadratic detrend (replaces statsmodels detrend(order=2))."""
    t = np.arange(len(x), dtype=float)
    return x - np.polyval(np.polyfit(t, x, 2), t)


def _next_pow2_ge(n: int) -> int:
    """Next power of two >= n."""
    return 1 << max(0, (n - 1).bit_length())


def _disper(w, h, g=9.81):
    """
    Wave number k from the linear dispersion relation (Goda 2010 iteration).

    Parameters
    ----------
    w : array_like  Angular frequency (rad/s).
    h : float       Depth (m).
    g : float       Gravity (m/s²).

    Returns
    -------
    k : ndarray  Wave number (1/m).
    """
    w   = np.atleast_1d(w).astype(float)
    h   = np.atleast_1d(h).astype(float)
    k0  = w ** 2 / g
    k0h = k0 * h
    kh  = np.where(k0h >= 1, k0h, np.sqrt(k0h))
    for _ in range(3):
        kh -= (kh - k0h / np.tanh(kh)) / (1 + k0h * (np.tanh(kh) ** -2 - 1))
    k        = kh / h
    k[w == 0] = 0.0
    k[np.isnan(k)] = 0.0
    return k.flatten()


def _CDF_by_bins(x, nbins):
    """Empirical CDF via bin counting."""
    x = np.asarray(x)
    xmin, xmax = x.min(), x.max()
    if xmin == xmax or nbins < 1:
        return np.array([0.0, 1.0]), np.array([xmin, xmax])
    edges      = np.linspace(xmin, xmax, nbins + 1)
    binCenters = 0.5 * (edges[:-1] + edges[1:])
    c          = np.array([np.mean(x <= e) for e in edges[1:]])
    c[-1]      = 1.0
    return c, binCenters


# ── spectral core ─────────────────────────────────────────────────────────────

def _spectral_helper(f, Sp, h, IG_divide=1/26, rho=1023, g=9.81):
    """Band-integrated spectral statistics as a dict-of-dicts."""
    f  = np.asarray(f)
    Sp = np.asarray(Sp)

    # Frequency bands
    IG_divide = list(np.ravel(IG_divide))
    if len(IG_divide) == 1:
        cut            = float(IG_divide[0])
        freq_bins      = np.array([[0, np.inf], [0, cut], [cut, np.inf]])
        freq_bin_names = ["Total", "IG", "SS"]
    elif len(IG_divide) == 2:
        lo, hi         = sorted(map(float, IG_divide))
        freq_bins      = np.array([[0, np.inf], [0, lo], [lo, hi], [hi, np.inf]])
        freq_bin_names = ["Total", "VL", "IG", "SS"]
    else:
        raise ValueError("IG_divide must be a float or a 1- or 2-element sequence.")

    # Depth-dependent quantities
    if h != 0:
        k  = _disper(2 * np.pi * f, h, g)
        kd = k * h
        with np.errstate(invalid="ignore", divide="ignore"):
            n = 0.5 + kd / np.sinh(2 * kd)
        n  = np.where(np.isnan(n), 1.0, n)
        C  = np.sqrt(g / k * np.tanh(kd))
        Cg = n * C

    # Spectral moments
    m0  = trapezoid(Sp, f)
    m1  = trapezoid(f * Sp, f)
    mn1 = trapezoid(Sp / f, f)

    fp    = f[np.argmax(Sp)]
    Tp    = 1.0 / fp
    Tmean = m0 / m1 if m1 > 0 else np.nan
    fmean = 1.0 / Tmean if Tmean and Tmean > 0 else np.nan
    fcent = np.sum(f * Sp) / np.sum(Sp) if np.sum(Sp) > 0 else np.nan
    Tcent = 1.0 / fcent if fcent and fcent > 0 else np.nan
    Tm10  = mn1 / m0 if m0 > 0 else np.nan

    # Band-integrated arrays
    SpTot = []; Hrms = []; Hsig = []
    E = []; F = []; Sxx = []; Ustokes = []

    for ii in range(len(freq_bin_names)):
        mask   = (f > freq_bins[ii, 0]) & (f <= freq_bins[ii, 1])
        Sp_int = trapezoid(Sp[mask], f[mask]) if mask.any() else 0.0
        SpTot.append(Sp_int)
        Hrms.append(np.sqrt(8.0 * Sp_int))
        Hsig.append(4.0 * np.sqrt(Sp_int))
        if h != 0:
            if mask.any():
                E.append(trapezoid(rho * g * Sp[mask], f[mask]))
                F.append(trapezoid(rho * g * Cg[mask] * Sp[mask], f[mask]))
                Sxx.append(trapezoid(rho * g * (2 * n[mask] - 0.5) * Sp[mask], f[mask]))
                Ustokes.append(trapezoid(g * Sp[mask] / (C[mask] * h), f[mask]))
            else:
                E.append(0.0); F.append(0.0); Sxx.append(0.0); Ustokes.append(0.0)

    results = {
        "h":                  {"data": np.array(h),             "long_name": "water depth",                    "units": "m"},
        "Sp":                 {"data": np.array(Sp),            "long_name": "Power Spectral Density",         "units": "m^2/Hz"},
        "f":                  {"data": np.array(f),             "long_name": "Frequency",                      "units": "Hz"},
        "Tp":                 {"data": np.array(Tp),            "long_name": "Peak Period",                    "units": "s"},
        "fp":                 {"data": np.array(fp),            "long_name": "Peak Frequency",                 "units": "Hz"},
        "Tmean":              {"data": np.array(Tmean),         "long_name": "Mean Period",                    "units": "s"},
        "fmean":              {"data": np.array(fmean),         "long_name": "Mean Frequency",                 "units": "Hz"},
        "Tcent":              {"data": np.array(Tcent),         "long_name": "Centroid Period",                "units": "s"},
        "fcent":              {"data": np.array(fcent),         "long_name": "Centroid Frequency",             "units": "Hz"},
        "Tenergy":            {"data": np.array(Tm10),          "long_name": "Energy Period (Tm-1,0)",         "units": "s"},
        "fenergy":            {"data": np.array(1/Tm10 if Tm10 and Tm10 > 0 else np.nan),
                                                                "long_name": "Energy Frequency",               "units": "Hz"},
        "SpTot":              {"data": np.array(SpTot),         "long_name": "Band-Integrated Spectral Energy","units": "m^2"},
        "Hrms":               {"data": np.array(Hrms),          "long_name": "Root Mean Square Wave Height",   "units": "m"},
        "Hsig":               {"data": np.array(Hsig),          "long_name": "Significant Wave Height",        "units": "m"},
        "freq_bin_names":     {"data": np.array(freq_bin_names),"long_name": "Frequency Band Names",           "units": "-"},
        "freq_bins":          {"data": np.array(freq_bins),     "long_name": "Frequency Band Boundaries",      "units": "Hz"},
        "IG_SS_freq_cutoff":  {"data": np.array(IG_divide),     "long_name": "IG/SS Frequency Cutoff(s)",      "units": "Hz"},
    }

    if h != 0:
        results.update({
            "E":       {"data": np.array(E),       "long_name": "Wave Energy",             "units": "J/m^2"},
            "F":       {"data": np.array(F),        "long_name": "Wave Energy Flux",        "units": "W/m"},
            "Sxx":     {"data": np.array(Sxx),      "long_name": "Radiation Stress",        "units": "N/m"},
            "Ustokes": {"data": np.array(Ustokes),  "long_name": "Stokes Drift (scaled)",   "units": "m/s"},
        })

    return results


def _exceedence(twl):
    """Runup exceedance statistics."""
    twl = np.asarray(twl)
    c, binCenters = _CDF_by_bins(twl, 20)
    twl2          = np.interp(0.98, c, binCenters)
    peaks, _      = find_peaks(twl, prominence=np.std(twl) / 3.0, distance=3)

    if len(peaks) >= 10:
        R2 = np.percentile(twl[peaks], 98)
    elif len(peaks) > 0:
        R2 = np.max(twl[peaks])
    else:
        R2 = np.percentile(twl, 98)

    return {
        "TWL_2pct":          {"data": R2,                    "long_name": "2% Total Water Level from Peaks",       "units": "m"},
        "TWL_2pct_not_peaks":{"data": twl2,                  "long_name": "2% Total Water Level from Full Series", "units": "m"},
        "Swash_2pct":        {"data": R2 - np.mean(twl),     "long_name": "2% Swash Excursion",                    "units": "m"},
        "TWL_mean":          {"data": np.mean(twl),          "long_name": "Mean Total Water Level",                "units": "m"},
    }


# ── public API ────────────────────────────────────────────────────────────────

def wave_stats(twl, dt, h=0, IG_divide=1/26, rho=1023, g=9.81, nperseg=None):
    """
    Compute wave statistics from a water-level time series.

    Parameters
    ----------
    twl      : array_like  Water level time series (m).
    dt       : float       Time step (s).
    h        : float       Water depth (m). Default 0 (skips depth-dependent terms).
    IG_divide: float or (float, float)
                           IG/SS frequency cutoff(s) in Hz. Default 1/26.
    rho      : float       Water density (kg/m³). Default 1023.
    g        : float       Gravity (m/s²). Default 9.81.
    nperseg  : int         Welch segment length. Chosen automatically if None.

    Returns
    -------
    dict
        {name: {'data': np.ndarray | scalar, 'long_name': str, 'units': str}, ...}
        Compatible with the original coastal_hazards_instrumentation wave_stats output.
    """
    twl = np.asarray(twl)
    if twl.ndim != 1 or twl.size < 2:
        raise ValueError("wave_stats expects a 1-D time series with length >= 2.")

    eta = twl - np.mean(twl)

    if nperseg is None:
        window_length = max(8, int(np.floor(len(eta) / 8)) - 1)
        nperseg       = min(_next_pow2_ge(window_length), len(eta))
    noverlap = min(int(nperseg * 0.75), nperseg - 1)

    f, Sp = scipy.signal.welch(
        eta,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=_quad_detrend,
    )
    if f.size > 0 and f[0] == 0:
        f  = f[1:]
        Sp = Sp[1:]

    return {**_spectral_helper(f, Sp, h, IG_divide=IG_divide, rho=rho, g=g),
            **_exceedence(twl)}
