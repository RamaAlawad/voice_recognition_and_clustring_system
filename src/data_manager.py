#=============================================================FIRST STATION==========================================================
#Data Management Module
import os
import json
import pickle
import pandas as pd
from pathlib import Path

class DataManager:
    def __init__(self, project_root="./"):
        """Initialize data manager"""
        self.project_root = Path(project_root)
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'LibriSpeech/test-clean/person1',
            'LibriSpeech/test-clean/person2', 
            'LibriSpeech/test-clean/person3',
            'data/processed',
            'models',
            'results',
            'uploads'  # For new voice samples to test
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
    
    def check_training_data(self):
     
        base_dir = self.project_root / 'LibriSpeech' / 'test-clean'
        
        if not base_dir.exists():
            return {'exists': False, 'message': 'Training data directory not found'}
        
        persons_info = {}
        total_files = 0
        
        for person_dir in ['person1', 'person2', 'person3']:
            person_path = base_dir / person_dir
            
            if person_path.exists():
                # Count audio files
                audio_files = []
                for ext in ['*.flac', '*.wav', '*.mp3']:
                    audio_files.extend(person_path.glob(f'**/{ext}'))
                
                persons_info[person_dir] = {
                    'files_count': len(audio_files),
                    'files': [f.name for f in audio_files[:5]]  # Show first 5 files
                }
                total_files += len(audio_files)
            else:
                persons_info[person_dir] = {
                    'files_count': 0,
                    'files': []
                }
        
        return {
            'exists': True,
            'total_files': total_files,
            'persons': persons_info,
            'ready_for_training': total_files > 0
        }
    
    def save_training_data(self, features, speaker_labels, file_paths, filename="training_data.pkl"):
        """Save processed training data"""
        data = {
            'features': features,
            'speaker_labels': speaker_labels,
            'file_paths': file_paths,
            'metadata': {
                'n_files': len(features),
                'n_speakers': len(set(speaker_labels)),
                'feature_dim': len(features[0]) if features else 0,
                'speakers': list(set(speaker_labels))
            }
        }
        
        filepath = self.project_root / 'data' / 'processed' / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Training data saved to {filepath}")
        return filepath
    
    def load_training_data(self, filename="training_data.pkl"):
        """Load processed training data"""
        filepath = self.project_root / 'data' / 'processed' / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def save_uploaded_file(self, uploaded_file, filename=None):
        """Save uploaded audio file for testing"""
        upload_dir = self.project_root / 'uploads'
        upload_dir.mkdir(exist_ok=True)
        
        if filename is None:
            filename = uploaded_file.name
        
        filepath = upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())
        
        return str(filepath)
    
    def get_uploaded_files(self):
        """Get list of uploaded files for testing"""
        upload_dir = self.project_root / 'uploads'
        
        if not upload_dir.exists():
            return []
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        files = []
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                })
        
        return files
    
    def create_sample_structure_info(self):
        """Create info about expected directory structure"""
        structure_info = {
            "description": "Voice Recognition System - Training Data Structure",
            "required_structure": {
                "LibriSpeech/test-clean/": {
                    "person1/": "Audio files for person 1 (any format: .flac, .wav, .mp3)",
                    "person2/": "Audio files for person 2", 
                    "person3/": "Audio files for person 3"
                }
            },
            "usage_flow": [
                "1. Place training audio files in person directories",
                "2. Run training to create voice recognition model",
                "3. Upload new voice samples to test recognition"
            ],
            "file_formats": ["FLAC", "WAV", "MP3"],
            "note": "Files can be in subdirectories within each person folder"
        }
        
        filepath = self.project_root / 'LibriSpeech' / 'structure_info.json'
        with open(filepath, 'w') as f:
            json.dump(structure_info, f, indent=2)
        
        return filepath
