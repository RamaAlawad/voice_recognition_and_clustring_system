import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
import pandas as pd

class RecognitionEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_training(self, true_labels, predicted_labels):
        """Evaluate training performance"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None, labels=list(set(true_labels))
        )
        
        # Per-class metrics
        speakers = list(set(true_labels))
        per_speaker_metrics = {}
        
        for i, speaker in enumerate(speakers):
            per_speaker_metrics[speaker] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=speakers)
        
        self.results = {
            'accuracy': accuracy,
            'per_speaker_metrics': per_speaker_metrics,
            'confusion_matrix': cm,
            'speakers': speakers
        }
        
        return self.results
    
    def print_evaluation_report(self):
        """Print detailed evaluation report"""
        if not self.results:
            print("No evaluation results available")
            return
        
        print("\n" + "="*50)
        print("VOICE RECOGNITION EVALUATION REPORT")
        print("="*50)
        
        print(f"Overall Accuracy: {self.results['accuracy']:.2%}")
        print(f"Number of Speakers: {len(self.results['speakers'])}")
        
        print("\nPer-Speaker Performance:")
        print("-" * 40)
        
        for speaker, metrics in self.results['per_speaker_metrics'].items():
            print(f"\n{speaker}:")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall:    {metrics['recall']:.2%}")
            print(f"  F1-Score:  {metrics['f1_score']:.2%}")
            print(f"  Samples:   {metrics['support']}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if not self.results:
            return None
        
        cm = self.results['confusion_matrix']
        speakers = self.results['speakers']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=speakers, yticklabels=speakers)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Speaker')
        plt.ylabel('True Speaker')
        
        return plt.gcf()
    
    def plot_recognition_results(self, predictions, confidences, filenames):
        """
        Plot recognition results for uploaded files
        
        Args:
            predictions: List of predicted speakers
            confidences: List of confidence dictionaries
            filenames: List of filenames
        """
        # Create DataFrame for easier plotting
        plot_data = []
        for i, (pred, conf_dict, filename) in enumerate(zip(predictions, confidences, filenames)):
            for speaker, confidence in conf_dict.items():
                plot_data.append({
                    'File': filename,
                    'Speaker': speaker,
                    'Confidence': confidence,
                    'Predicted': speaker == pred
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create confidence plot
        fig = px.bar(df, x='File', y='Confidence', color='Speaker',
                     title='Recognition Confidence Scores',
                     text='Confidence')
        
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig

