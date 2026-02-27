import numpy as np
import matplotlib.pyplot as plt
import os

FEATURES_DIR = os.path.join("data", "features")
feature_file = os.path.join(FEATURES_DIR, "test_tone.npy")

# Load the mel spectrogram
mel = np.load(feature_file)

# Plot
plt.figure(figsize=(10, 4))
plt.imshow(mel, origin='lower', aspect='auto', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel-spectrogram of test_tone.wav")
plt.xlabel("Frames")
plt.ylabel("Mel bins")
plt.tight_layout()

# Save figure
output_path = "mel_preview.png"
plt.savefig(output_path)
plt.show()
print(f"Saved mel spectrogram preview as {output_path}")
