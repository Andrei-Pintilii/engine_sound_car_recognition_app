import os
from src.utils_audio import load_audio, save_wav
from src.config import RAW_DIR, PROCESSED_DIR, SAMPLE_RATE, CLIP_DURATION

def preprocess_all():
    # Ensure the main processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Iterate through each class folder (1 through 8)
    for class_id in range(1, 9):
        # Construct path to the class subfolder
        class_raw_dir = os.path.join(RAW_DIR, str(class_id))
        
        # Skip if the folder doesn't exist (e.g., if you haven't organized all classes yet)
        if not os.path.isdir(class_raw_dir):
            continue
            
        # Create a matching subfolder in the processed directory
        class_out_dir = os.path.join(PROCESSED_DIR, str(class_id))
        os.makedirs(class_out_dir, exist_ok=True)
        
        # Get all wav files in this specific class folder
        files = [f for f in os.listdir(class_raw_dir) if f.lower().endswith('.wav')]
        print(f"Processing {len(files)} files for Class {class_id}...")

        for f in files:
            src = os.path.join(class_raw_dir, f)
            try:
                # Load and resample audio
                y = load_audio(src, sr=SAMPLE_RATE, duration=CLIP_DURATION)
                
                # Save to the corresponding processed subfolder
                out = os.path.join(class_out_dir, f)
                save_wav(out, y, SAMPLE_RATE)
                # print("Processed ->", out) # Optional: keep commented to reduce terminal noise
            except Exception as e:
                print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    preprocess_all()
    print("Pre-processing of all classes complete.")