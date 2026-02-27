import soundfile as sf
import numpy as np
import librosa

def load_audio(path, sr=16000, duration=3.0):
    y, orig_sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y.astype(float), orig_sr=orig_sr, target_sr=sr)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)

def save_wav(path, y, sr=16000):
    sf.write(path, y, sr)
