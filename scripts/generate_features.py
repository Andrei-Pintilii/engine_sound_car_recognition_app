import os
import numpy as np
import soundfile as sf
from src.feature_extractor import compute_log_mel
from src.config import PROCESSED_DIR, FEATURES_DIR, FEATURE_EXT

def generate_all():
    # Ensure the features output directory exists
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    # Iterate through folders 1 to 8 in the processed directory
    for class_id in range(1, 9):
        class_processed_dir = os.path.join(PROCESSED_DIR, str(class_id))
        
        # Skip if the class folder doesn't exist
        if not os.path.isdir(class_processed_dir):
            continue
            
        print(f"Generating features for Class {class_id}...")
        
        # Get all processed wav files for this class
        files = [f for f in os.listdir(class_processed_dir) if f.lower().endswith('.wav')]
        
        for i, f in enumerate(files):
            src = os.path.join(class_processed_dir, f)
            
            try:
                # 1. Load the audio file
                y, sr = sf.read(src)
                
                # 2. Convert to Mono if necessary
                if y.ndim > 1:
                    y = y.mean(axis=1)
                
                # 3. Extract Log-Mel Spectrogram (ensure N_MELS=128 in your config/extractor)
                mel = compute_log_mel(y, sr=sr)
                
                # 4. Standardize output filename to "Class (Instance).npy"
                # This ensures the training script can extract labels from filenames
                out_name = f"{class_id} ({i+1}){FEATURE_EXT}"
                out_path = os.path.join(FEATURES_DIR, out_name)
                
                # 5. Save the feature array
                np.save(out_path, mel)
                # print("Saved feature:", out_path)
                
            except Exception as e:
                print(f"Error extracting features from {f}: {e}")

if __name__ == "__main__":
    generate_all()
    print("\nFeature extraction complete for all classes.")