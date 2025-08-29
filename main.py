"""
Main Application - Voice Recognition System
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.audio_processor import AudioProcessor
from src.voice_recognizer import VoiceRecognizer
from src.evaluator import RecognitionEvaluator
from src.data_manager import DataManager

def main():
    """Main pipeline for voice recognition system"""
    print("="*60)
    print("VOICE RECOGNITION SYSTEM")
    print("="*60)
    
    # Initialize components
    data_manager = DataManager()
    audio_processor = AudioProcessor()
    recognizer = VoiceRecognizer(model_type='random_forest')
    evaluator = RecognitionEvaluator()
    
    # Create structure info
    data_manager.create_sample_structure_info()
    
    # Check training data
    training_info = data_manager.check_training_data()
    
    if not training_info['exists'] or not training_info['ready_for_training']:
        print("\n❌ Training data not found!")
        print("Please organize your training data as follows:")
        print("LibriSpeech/test-clean/")
        print("  ├── person1/  (audio files for person 1)")
        print("  ├── person2/  (audio files for person 2)")
        print("  └── person3/  (audio files for person 3)")
        print("\nSupported formats: .flac, .wav, .mp3")
        return
    
    print("\n✅ Training data found!")
    for person, info in training_info['persons'].items():
        print(f"  {person}: {info['files_count']} files")
    
    # Check if we have processed training data
    existing_data = data_manager.load_training_data()
    
    if existing_data is None:
        # Process training data
        print("\n" + "-"*40)
        print("PROCESSING TRAINING DATA")
        print("-"*40)
        
        features, speaker_labels, file_paths = audio_processor.process_training_data(
            "LibriSpeech/test-clean"
        )
        
        # Save processed data
        data_manager.save_training_data(features, speaker_labels, file_paths)
    else:
        print("\n✅ Using existing processed training data")
        features = existing_data['features']
        speaker_labels = existing_data['speaker_labels']
        file_paths = existing_data['file_paths']
    
    # Train the model
    print("\n" + "-"*40)
    print("TRAINING RECOGNITION MODEL")
    print("-"*40)
    
    train_accuracy = recognizer.train(features, speaker_labels)
    
    # Save trained model
    recognizer.save_model("models/voice_recognizer.pkl")
    
    # Evaluate on training data
    predictions, confidences = recognizer.predict_multiple(features)
    evaluation_results = evaluator.evaluate_training(speaker_labels, predictions)
    evaluator.print_evaluation_report()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Model saved to: models/voice_recognizer.pkl")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print("\nNow you can upload new voice samples for recognition!")

if __name__ == "__main__":
    main()