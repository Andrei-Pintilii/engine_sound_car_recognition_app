import numpy as np
import soundfile as sf
import os

# Create folders if they don't exist
raw_dir = os.path.join("data", "raw")
os.makedirs(raw_dir, exist_ok=True)

# Audio settings
sr = 16000        # 16 kHz
duration = 3.0    # 3 seconds
t = np.linspace(0, duration, int(sr*duration), endpoint=False)

# Generate a test tone resembling engine harmonics
y = 0.5*np.sin(2*np.pi*120*t) + 0.3*np.sin(2*np.pi*600*t)

# Save the file
output_path = os.path.join(raw_dir, "test_tone.wav")
sf.write(output_path, y.astype("float32"), sr)

print(f"Created {output_path}")
