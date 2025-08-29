"""
Voice Recognition Module
Handles training and recognition of speakers
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

class VoiceRecognizer:
    def __init__(self, model_type='random_forest'):
        """
        Initialize voice recognizer
        
        Args:
            model_type: Type of classifier ('random_forest', 'svm', 'knn')
        """
        self.model_type = model_type
        self.model = None
        self.speaker_labels = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError("model_type must be 'random_forest', 'svm', or 'knn'")
    
    def train(self, features, speaker_labels):
        """
        Train the voice recognition model
        
        Args:
            features: List of feature vectors
            speaker_labels: List of corresponding speaker labels
        """
        features_array = np.array(features)
        self.speaker_labels = list(set(speaker_labels))
        
        print(f"Training {self.model_type} model...")
        print(f"Features shape: {features_array.shape}")
        print(f"Speakers: {self.speaker_labels}")
        
        # Train the model
        self.model.fit(features_array, speaker_labels)
        self.is_trained = True
        
        # Evaluate on training data (for reference)
        train_predictions = self.model.predict(features_array)
        train_accuracy = accuracy_score(speaker_labels, train_predictions)
        
        print(f"Training completed! Training accuracy: {train_accuracy:.2%}")
        
        return train_accuracy
    
    def predict_speaker(self, features, return_confidence=True):
        """
        Predict speaker for given features
        
        Args:
            features: Feature vector for audio sample
            return_confidence: Whether to return confidence scores
            
        Returns:
            tuple: (predicted_speaker, confidence_dict) if return_confidence=True
            str: predicted_speaker if return_confidence=False
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        features_array = np.array(features).reshape(1, -1)
        
        # Get prediction
        predicted_speaker = self.model.predict(features_array)[0]
        
        if return_confidence:
            # Get confidence scores for all classes
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                confidence_dict = dict(zip(self.model.classes_, probabilities))
            else:
                # For models without probability support
                confidence_dict = {predicted_speaker: 1.0}
            
            return predicted_speaker, confidence_dict
        
        return predicted_speaker
    
    def predict_multiple(self, features_list):
        """
        Predict speakers for multiple audio samples
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        features_array = np.array(features_list)
        predictions = self.model.predict(features_array)
        
        confidence_scores = []
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_array)
            for i, probs in enumerate(probabilities):
                confidence_dict = dict(zip(self.model.classes_, probs))
                confidence_scores.append(confidence_dict)
        else:
            confidence_scores = [{pred: 1.0} for pred in predictions]
        
        return predictions, confidence_scores
    
    def evaluate(self, test_features, test_labels):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        test_features_array = np.array(test_features)
        predictions = self.model.predict(test_features_array)
        
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        return accuracy, report
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'speaker_labels': self.speaker_labels,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.speaker_labels = model_data['speaker_labels']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
        print(f"Known speakers: {self.speaker_labels}")
