import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import pandas as pd
import seaborn as sns
import os
import re

# --- Configuration ---
# 1. Set the path to the folder containing your Mel spectrogram files
FEATURES_FOLDER = 'C:\\Users\\pinti\\Desktop\\proiect_TIA\\data\\features'

# 2. Recommended standard vehicle types for your 8 classes
# Update this dictionary later once you identify which number corresponds to which vehicle
CLASS_NAMES = {
    1: 'Class 1 (Bus)',
    2: 'Class 2 (Minibus)',
    3: 'Class 3 (Pickup)',
    4: 'Class 4 (Sports Car)',
    5: 'Class 5 (Jeep)',
    6: 'Class 6 (Truck)',
    7: 'Class 7 (Crossover)',
    8: 'Class 8 (Car)'
}

# --- Data Loading and Preprocessing ---
all_features = []
all_labels = []

print(f"Starting data aggregation from: {FEATURES_FOLDER}")

# Regular expression to extract the vehicle type id (the class label)
# It looks for a number at the beginning of the filename, before the first space
# Example: '1 (45).npy' -> extracts '1'
filename_pattern = re.compile(r'^(\d+)\s')

# Check if the folder exists
if not os.path.isdir(FEATURES_FOLDER):
    print(f"Error: The folder '{FEATURES_FOLDER}' was not found.")
    print("Please make sure your folder containing the .npy files is named 'features' or update the FEATURES_FOLDER variable.")
else:
    # Iterate over all files in the features folder
    for filename in os.listdir(FEATURES_FOLDER):
        if filename.endswith('.npy'):
            filepath = os.path.join(FEATURES_FOLDER, filename)
            
            # 1. Extract Label from filename
            match = filename_pattern.match(filename)
            if match:
                class_label = int(match.group(1))
                if class_label in CLASS_NAMES:
                    all_labels.append(class_label)
                    
                    # 2. Load Feature (Mel Spectrogram)
                    try:
                        feature = np.load(filepath)
                        all_features.append(feature)
                    except Exception as e:
                        print(f"Could not load or process file {filename}: {e}")
                else:
                    print(f"Skipping file {filename}: Class ID {class_label} not in range 1-8.")
            else:
                print(f"Skipping file {filename}: Filename does not match expected pattern 'ID (number).npy'.")

# Consolidate into NumPy arrays
if not all_features:
    print("No features were successfully loaded. Exiting.")
    exit()

features = np.array(all_features)
labels = np.array(all_labels)

# Flatten the Mel Spectrograms: (N, H, W) -> (N, H*W)
N_samples = features.shape[0]
flattened_features = features.reshape(N_samples, -1)
print(f"Successfully loaded {N_samples} samples.")
print(f"Data shape for t-SNE: {flattened_features.shape}")

# --- t-SNE Dimensionality Reduction ---
print("Starting t-SNE dimensionality reduction (this may take a few minutes)...")
time_start = time.time()

# Note: Adjust 'perplexity' (commonly 5 to 50) and 'n_iter' if results look poor
tsne = TSNE(n_components=2, 
            verbose=1, 
            perplexity=30, 
            max_iter=3000, 
            random_state=42,
            n_jobs=-1) # Use all available cores

# Fit and transform the data
tsne_results = tsne.fit_transform(flattened_features)

print(f't-SNE completed in {time.time()-time_start:.2f} seconds.')

# --- Plotting the Results ---

# Prepare data for plotting
df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE Dimension 1', 't-SNE Dimension 2'])
df_tsne['Class'] = labels.astype(int)
df_tsne['Vehicle Type'] = df_tsne['Class'].map(CLASS_NAMES)

plt.figure(figsize=(12, 10))

# Create the scatter plot
sns.scatterplot(
    x='t-SNE Dimension 1', y='t-SNE Dimension 2',
    hue='Vehicle Type',
    data=df_tsne,
    legend='full',
    alpha=0.8,
    palette=sns.color_palette("tab10", len(CLASS_NAMES))
)

plt.title('t-SNE Plot of Vehicle Sound Mel Spectrograms (Colored by Class ID)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show() 

print("\n*** Analysis Guide ***")
print("Once the plot appears, visually inspect the clusters:")
print("* **Well-Separated Clusters:** Indicates that your 8 classes have distinct sound characteristics (Mel spectrograms). This is a great sign for your classification project.")
print("* **Overlapping Clusters:** Suggests that the sounds of those vehicle types are too similar based on your Mel spectrogram extraction, or that the classes are acoustically ambiguous.")