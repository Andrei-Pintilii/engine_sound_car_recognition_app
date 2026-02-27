import os
import shutil

# Path where all your .wav files are currently located
SOURCE_DIR = 'C:\\Users\\pinti\\Desktop\\proiect_TIA\\data\\raw' 

def organize_files():
    # Get list of all files in the directory
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]
    
    if not all_files:
        print("No .wav files found in the source directory.")
        return

    for filename in all_files:
        # Extract class ID (the first character of the filename)
        class_id = filename[0]
        
        # Check if it's a valid class digit (1-8)
        if class_id.isdigit() and '1' <= class_id <= '8':
            target_folder = os.path.join(SOURCE_DIR, class_id)
            
            # Create the subfolder if it doesn't exist
            os.makedirs(target_folder, exist_ok=True)
            
            # Move the file
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(target_folder, filename)
            shutil.move(src_path, dst_path)

if __name__ == "__main__":
    print("Organizing files into class folders (1-8)...")
    organize_files()
    print("Organization complete.")