import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Run this from C:\Users\pinti\Desktop\proiect_TIA
BUS_RAW_DIR = os.path.join('data', 'raw', '1')

def augment_bus_class():
    files = [f for f in os.listdir(BUS_RAW_DIR) if f.endswith('.wav') and 'aug' not in f]
    print(f"Augmenting {len(files)} Bus files to fix Class 1 Recall...")

    for filename in tqdm(files):
        path = os.path.join(BUS_RAW_DIR, filename)
        try:
            y, sr = librosa.load(path, sr=None)
            base = os.path.splitext(filename)[0]

            # Variation A: Pitch Shift Down (Simulates heavy engine load)
            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
            sf.write(os.path.join(BUS_RAW_DIR, f"{base}_aug_low.wav"), y_pitch, sr)

            # Variation B: Adding "Road Hum" Noise
            noise = np.random.randn(len(y))
            y_noise = y + 0.008 * noise
            sf.write(os.path.join(BUS_RAW_DIR, f"{base}_aug_noise.wav"), y_noise, sr)

        except Exception as e:
            print(f"Error on {filename}: {e}")

if __name__ == "__main__":
    augment_bus_class()