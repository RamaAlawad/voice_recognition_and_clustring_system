# After your uploading your data come to this to process and clean it
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def load_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None
    
    def preprocess_audio(self, audio):
        """
        Apply preprocessing to audio
        - Normalize volume
        - Remove silence
        """
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Remove leading and trailing silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio
    
    def extract_mfcc_features(self, audio):
        """Extract MFCC features from audio"""
        try:
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc
            )
            
            # Take mean across time frames
            mfcc_mean = np.mean(mfcc, axis=1)
            
            return mfcc_mean
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def process_audio_file(self, audio_path):
        """Complete processing pipeline for a single audio file"""
        # Load audio
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # Preprocess
        audio = self.preprocess_audio(audio)
        
        # Extract features (MFCC are already embeddings/features)
        features = self.extract_mfcc_features(audio)
        
        return features
    
    def process_training_data(self, base_dir):
        """
        Process training data from LibriSpeech structure
        
        Args:
            base_dir: Path to LibriSpeech/test-clean directory
            
        Returns:
            tuple: (features_list, speaker_labels_list, file_paths_list)
        """
        base_path = Path(base_dir)
        all_features = []
        all_speakers = []
        all_file_paths = []
        
        print("Processing training data...")
        
        # Process each person directory (person1, person2, person3)
        for person_dir in base_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            print(f"Processing {person_name}...")
            
            # Process all audio files in this person's directory
            audio_files = []
            for ext in ['*.flac', '*.wav', '*.mp3']:
                audio_files.extend(person_dir.glob(f'**/{ext}'))
            
            person_features = []
            person_file_paths = []
            
            for audio_file in audio_files:
                features = self.process_audio_file(audio_file)
                if features is not None:
                    person_features.append(features)
                    person_file_paths.append(str(audio_file))
            
            print(f"  Processed {len(person_features)} files for {person_name}")
            
            # Add to global lists
            all_features.extend(person_features)
            all_speakers.extend([person_name] * len(person_features))
            all_file_paths.extend(person_file_paths)
        
        print(f"Total training data: {len(all_features)} files from {len(set(all_speakers))} speakers")
        return all_features, all_speakers, all_file_paths

