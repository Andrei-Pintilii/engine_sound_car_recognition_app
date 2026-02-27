import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# Audio settings
SAMPLE_RATE = 16000        # 16 kHz
CLIP_DURATION = 3.0        # seconds
N_MELS = 128               # mel bins
HOP_LENGTH = 256
N_FFT = 1024

# Feature file format
FEATURE_EXT = ".npy"
