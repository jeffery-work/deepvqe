import numpy as np
from pesq import pesq
from pystoi.stoi import stoi
import torch
import torchaudio.functional as F


def SI_SDR(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    estimation, reference = np.broadcast_arrays(estimation, reference) # align array dimension
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
                      / reference_energy # axis (0,length-1) map to (-length, -1)

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)

def STOI(ref, est, fs=16000):
    if fs is not 16000:
        ref = F.resample(ref, fs, 16000)
        est = F.resample(est, fs, 16000)

    return stoi(ref, est, fs, extended=False)


def PESQ(ref, est, fs=16000):
    return pesq(fs, ref, est, "wb")