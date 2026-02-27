import numpy as np
import librosa
from .config import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH

def compute_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)
