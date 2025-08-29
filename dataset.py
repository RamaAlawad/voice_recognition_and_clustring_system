import requests
import os
import random
import pandas as pd
from pathlib import Path
import time

# Download settings
save_folder = "data/raw"
num_recordings = 100
github_base_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/raw/master/recordings/"

# Create save directory
Path(save_folder).mkdir(exist_ok=True)

# List of possible digits, speakers, and repetitions
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
speakers = ['jackson', 'nicolas', 'theo', 'yweweler']
repetitions = [str(i) for i in range(20)]  # 0 to 19

# Generate all possible filenames
all_possible_files = []
for digit in digits:
    for speaker in speakers:
        for repetition in repetitions:
            filename = f"{digit}_{speaker}_{repetition}.wav"
            all_possible_files.append(filename)

# Select 100 random files to try downloading
files_to_download = random.sample(all_possible_files, min(num_recordings * 2, len(all_possible_files)))

metadata = []
downloaded_count = 0

print(f"Attempting to download {num_recordings} files from GitHub...")

for i, filename in enumerate(files_to_download):
    if downloaded_count >= num_recordings:
        break
        
    try:
        file_url = github_base_url + filename
        response = requests.get(file_url, timeout=10)
        
        if response.status_code == 200:
            # Save the file
            save_path = os.path.join(save_folder, filename)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Extract info from filename
            name_parts = filename.replace('.wav', '').split('_')
            
            if len(name_parts) >= 3:
                metadata.append({
                    'file_name': filename,
                    'digit': name_parts[0],
                    'speaker': name_parts[1],
                    'repetition': name_parts[2],
                    'download_url': file_url
                })
            
            downloaded_count += 1
            print(f"Downloaded {downloaded_count}/{num_recordings}: {filename}")
            
            # Add small delay to be polite to the server
            time.sleep(0.1)
            
        else:
            print(f"File not found (skipping): {filename}")
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Save metadata to CSV file
if metadata:
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(save_folder, "metadata.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Metadata saved to: {csv_path}")
    
    # Show download statistics
    print(f"\nDownload Summary:")
    print(f"Total files successfully downloaded: {len(metadata)}")
    if len(metadata) > 0:
        print(f"Digits distribution: {df['digit'].value_counts().to_dict()}")
        print(f"Speakers distribution: {df['speaker'].value_counts().to_dict()}")
else:
    print("No files were downloaded successfully")

print("Download process completed!")

# If we couldn't get enough files, show a message
if downloaded_count < num_recordings:
    print(f"\nNote: Only found {downloaded_count} files out of requested {num_recordings}.")
    print("This dataset might have fewer files than expected, or some files might not exist.")




import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import random

def merge_audio_files(audio_files, output_path, target_sr=22050):
    """Merge multiple audio files into one"""
    merged_audio = np.array([], dtype=np.float32)
    
    for audio_file in audio_files:
        try:
            # Load audio file
            audio, sr = librosa.load(audio_file, sr=target_sr)
            
            # Add a small silence between files (0.1 seconds)
            silence = np.zeros(int(0.1 * target_sr), dtype=np.float32)
            
            # Concatenate audio
            merged_audio = np.concatenate([merged_audio, audio, silence])
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return merged_audio

def merge_audio_in_groups(input_folder, output_folder, group_size=5):
    """Merge audio files in groups and save them"""
    
    # Create output directory
    Path(output_folder).mkdir(exist_ok=True)
    
    # Get all WAV files from input folder
    audio_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                  if f.endswith('.wav')]
    
    # Shuffle files for random grouping
    random.shuffle(audio_files)
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Creating groups of {group_size} files each")
    
    # Process files in groups
    group_count = 0
    for i in range(0, len(audio_files), group_size):
        group_files = audio_files[i:i + group_size]
        
        if len(group_files) < group_size:
            print(f"Skipping last group of {len(group_files)} files (less than {group_size})")
            break
        
        # Create output filename
        output_filename = f"merged_group_{group_count + 1}.wav"
        output_path = os.path.join(output_folder, output_filename)
        
        # Merge audio files
        print(f"Merging group {group_count + 1}: {[os.path.basename(f) for f in group_files]}")
        merged_audio = merge_audio_files(group_files, output_path)
        
        # Save merged audio
        sf.write(output_path, merged_audio, 22050)
        print(f"Saved: {output_filename} (duration: {len(merged_audio)/22050:.2f} seconds)")
        
        group_count += 1
    
    print(f"\nSuccessfully created {group_count} merged audio files in '{output_folder}'")

def main():
    input_folder = "data/raw"  # Folder with your 100 audio files
    output_folder = "data/merged_audio_groups"     # Output folder for merged files
    group_size = 5                            # Number of files to merge together
    
    # Run the merging process
    merge_audio_in_groups(input_folder, output_folder, group_size)

if __name__ == "__main__":
    main()