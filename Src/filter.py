
###### Baseline wander removal Filter #####



import numpy as np
from scipy.signal import butter, filtfilt, firwin
def remove_baseline_wander(ecg, fs=1000, cutoff=0.67):
    """
    Remove baseline wander using a high-pass FIR filter.
    Parameters
    ecg : ndarray
        ECG signal.
    fs : int
        Sampling frequency.
    cutoff : float
        High-pass cutoff frequency (Hz).
    Returns
    -------
    ndarray
        Filtered ECG signal.
    """
    taps = firwin(
        numtaps=1501,
        cutoff=cutoff,
        fs=fs,
        pass_zero=False
    )

    return filtfilt(taps, 1, ecg)





####### ECG pre-processing and display filter  #######


import numpy as np
from scipy import signal
import condat_tv

def ecg_preprocess(ecg_signal, fs=1000, hp_cutoff=0.67):
    """
    Preprocessing an ECG signal by performing:

    1. Missing sample interpolation (zero-value replacement)
    2. Baseline wander removal
    3. Low-pass filtering
    4. Signal smoothing using Total Variation (TV) denoising

    Parameters
    ----------
    ecg_signal : numpy.ndarray
        Input ECG signal.

    fs : int, optional
        Sampling frequency in Hz.
        Default is 1000 Hz.

    hp_cutoff : float, optional
        High-pass cutoff frequency for baseline removal.
        Default is 0.67 Hz.

    Returns
    -------
    numpy.ndarray
        Smoothed ECG signal.
    """

    ecg = np.asarray(ecg_signal, dtype=float).copy()

    # ------------------------------------------------------------------
    # Replacing (zero) samples with previous valid sample
    # ------------------------------------------------------------------
    for i in range(1, len(ecg)):
        if ecg[i] == 0:
            ecg[i] = ecg[i - 1]

    # ------------------------------------------------------------------
    # Baseline Wander Removing
    # ------------------------------------------------------------------
    fir_coeff = signal.firwin(
        numtaps=2377,
        cutoff=hp_cutoff,
        fs=fs,
        pass_zero=False,
        window="hamming",
    )

    baseline_removed = signal.filtfilt(fir_coeff, 1, ecg)

    # ------------------------------------------------------------------
    # Low-pass Filters
    # ------------------------------------------------------------------
    b_lp45, a_lp45 = signal.butter(2, 45, fs=fs, btype="low")
    b_lp15, a_lp15 = signal.butter(2, 15, fs=fs, btype="low")

    lowpass_45 = signal.lfilter(b_lp45, a_lp45, baseline_removed)
    lowpass_15 = signal.lfilter(b_lp15, a_lp15, baseline_removed)

    # ------------------------------------------------------------------
    # Sparse Component
    # ------------------------------------------------------------------
    sparse_component = lowpass_45 - lowpass_15

    # ------------------------------------------------------------------
    # Total Variation Denoising
    # ------------------------------------------------------------------
    denoised = condat_tv.tv_denoise(sparse_component, 6.5)

    # ------------------------------------------------------------------
    #  Smoothed ECG
    # ------------------------------------------------------------------
    smoothed_ecg = lowpass_15 + denoised

    return smoothed_ecg
