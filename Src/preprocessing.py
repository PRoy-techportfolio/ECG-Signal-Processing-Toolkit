##### pre processing functions for ECG ##########

import numpy as np
from scipy import signal
from scipy.signal import butter
from datetime import datetime
from pytz import timezone


# ======================================================================
# Missing Sample Correction
# ======================================================================

def replace_missing_samples(ecg_signal):
    """
    Replace missing (zero-valued) samples using the mean of
    adjacent samples.

    Parameters
    ----------
    ecg_signal : ndarray
        Input ECG signal.

    Returns
    -------
    ndarray
        Corrected ECG signal.
    """

    ecg = np.asarray(ecg_signal, dtype=float).copy()

    for i in range(1, len(ecg) - 1):

        if ecg[i] == 0:

            ecg[i] = (
                ecg[i - 1] + ecg[i + 1]
            ) / 2

    return ecg


# ======================================================================
# Timestamp Conversion
# ======================================================================

def timestamp_to_datetime(timestamp):
    """
    Convert a Unix timestamp into Indian Standard Time (IST).

    Parameters
    ----------
    timestamp : int or float

    Returns
    -------
    tuple
        (date, time)
    """

    ist_time = datetime.fromtimestamp(timestamp).astimezone(
        timezone("Asia/Kolkata")
    )

    date = ist_time.strftime("%d-%m-%Y")
    current_time = ist_time.strftime("%H:%M:%S")

    return date, current_time


# ======================================================================
# Bandpass Filter Design
# ======================================================================

def design_bandpass_filter(
    lowcut,
    highcut,
    fs=1000,
    order=4,
):
    """
    Design a Butterworth band-pass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency (Hz)

    highcut : float
        Upper cutoff frequency (Hz)

    fs : int
        Sampling frequency.

    order : int
        Butterworth filter order.

    Returns
    -------
    ndarray
        Second-order section (SOS) coefficients.
    """

    return butter(
        order,
        [lowcut, highcut],
        fs=fs,
        btype="bandpass",
        output="sos",
    )


# ======================================================================
# Rhythm Regularity Check
# ======================================================================

def check_rhythm_regularity(
    ecg_signal,
    fs=1000,
):
    """
    Determine whether an ECG rhythm is regular
    based on RR interval variability.

    Parameters
    ----------
    ecg_signal : ndarray

    fs : int

    Returns
    -------
    str
        "Regular Rhythm" or
        "Irregular Rhythm"
    """

    # ----------------------------------------------------------
    # Baseline Wander Removal
    # ----------------------------------------------------------

    fir = signal.firwin(
        1735,
        cutoff=3,
        fs=fs,
        pass_zero=False,
        window="hamming",
    )

    filtered = signal.filtfilt(
        fir,
        1,
        ecg_signal,
    )

    # ----------------------------------------------------------
    # Low-pass Filter
    # ----------------------------------------------------------

    b, a = signal.butter(
        15,
        35,
        fs=fs,
        btype="low",
    )

    filtered = signal.lfilter(
        b,
        a,
        filtered,
    )

    # ----------------------------------------------------------
    # Energy Signal
    # ----------------------------------------------------------

    energy = filtered ** 2

    threshold = np.max(energy[1000:5000]) / 4

    peaks, _ = signal.find_peaks(
        energy,
        distance=400,
        height=threshold,
    )

    rr_intervals = np.diff(peaks)

    if len(rr_intervals) == 0:

        return "Unknown Rhythm"

    rr_std = np.std(rr_intervals)

    if rr_std < 90:

        return "Regular Rhythm"

    return "Irregular Rhythm"
