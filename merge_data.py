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