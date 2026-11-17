"""WAV PCM -> tensor in memory for pyannote Community-1.

Bypasses torchcodec's AudioDecoder on Windows, preserving the See2Sound
Whisper audio, which is extracted as 44.1 kHz PCM 16-bit WAV.

The returned dict follows pyannote's public in-memory input contract:
    {'waveform': Tensor[channels, samples], 'sample_rate': int}
"""
from __future__ import annotations

from math import gcd
from pathlib import Path


def read_waveform_for_pyannote(audio_path: str | Path, sample_rate: int = 16000):
    """Read and resample one WAV without TorchCodec or FFmpeg shared DLLs.

    scipy and torch are already used by the See2Sound pipeline. The signal is
    mono float32 with finite values, in range [-1, 1], shape [1, N].
    """
    import numpy as np
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f'Audio WAV nao encontrado: {path}')
    if path.suffix.lower() != '.wav':
        raise ValueError('A diarizacao sem TorchCodec exige um arquivo WAV. ' 
                         f'Recebido: {path.suffix or "sem extensao"}.')

    rate, samples = wavfile.read(str(path))
    if rate <= 0 or sample_rate <= 0 or np.size(samples) == 0:
        raise ValueError(f'Audio WAV vazio ou taxa de amostragem invalida: {path}')

    # Convert stereo/multichannel WAV to mono after converting to float,
    # so summing signed integer samples never overflows.
    if samples.ndim not in (1, 2):
        raise ValueError(f'Formato WAV inesperado: dimensoes={samples.ndim}')

    if np.issubdtype(samples.dtype, np.unsignedinteger):
        info = np.iinfo(samples.dtype)
        center = (info.max + 1) / 2.0
        pcm = (samples.astype(np.float32) - center) / center
    elif np.issubdtype(samples.dtype, np.signedinteger):
        info = np.iinfo(samples.dtype)
        pcm = samples.astype(np.float32) / float(-info.min)
    elif np.issubdtype(samples.dtype, np.floating):
        pcm = samples.astype(np.float32)
    else:
        raise ValueError(f'Formato WAV nao suportado: {samples.dtype}')

    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1, dtype=np.float32)
    pcm = np.nan_to_num(pcm, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.clip(pcm, -1.0, 1.0)

    if int(rate) != int(sample_rate):
        factor = gcd(int(rate), int(sample_rate))
        pcm = resample_poly(pcm, sample_rate // factor, rate // factor)

    pcm = np.ascontiguousarray(np.clip(pcm, -1.0, 1.0), dtype=np.float32)
    if pcm.size == 0:
        raise ValueError(f'Audio vazio apos reamostragem: {path}')

    return {
        'waveform': torch.from_numpy(pcm).unsqueeze(0),
        'sample_rate': int(sample_rate),
    }
