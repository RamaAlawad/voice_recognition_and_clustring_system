#!/usr/bin/env python3
"""
Voice Clustering System - Main Implementation
Automatically clusters speakers in audio without prior knowledge
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import webrtcvad
import json
from datetime import timedelta
import argparse

# Try importing pyannote for better embeddings
try:
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Using basic MFCC features.")

class VoiceClusteringSystem:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        self.embeddings_model = None
        
        # Initialize speaker embedding model if available
        if PYANNOTE_AVAILABLE:
            try:
                self.pipeline = SpeakerDiarization.from_pretrained(
                    "pyannote/speaker-diarization", 
                    use_auth_token=None  # You may need HuggingFace token
                )
                print("Loaded pyannote.audio speaker diarization pipeline")
            except:
                print("Could not load pyannote pipeline. Using fallback method.")
                self.pipeline = None
        else:
            self.pipeline = None
    
    def process_audio_file(self, audio_path, output_dir="results"):
        """
        Main processing pipeline for audio file
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save results
            
        Returns:
            dict: Clustering results with speaker segments
        """
        print(f"Processing: {audio_path}")
        
        # Step 1: Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        print(f"Loaded audio: {len(audio)/sr:.1f} seconds")
        
        # Step 2: Voice Activity Detection
        speech_segments = self.detect_speech_segments(audio, sr)
        print(f"Found {len(speech_segments)} speech segments")
        
        # Step 3: Extract embeddings or use pyannote pipeline
        if self.pipeline:
            results = self.cluster_with_pyannote(audio_path)
        else:
            results = self.cluster_with_basic_method(speech_segments, audio, sr)
        
        # Step 4: Save results
        os.makedirs(output_dir, exist_ok=True)
        self.save_results(results, output_dir, Path(audio_path).stem)
        
        return results
    
    def detect_speech_segments(self, audio, sr, segment_duration=0.03):
        """
        Detect speech segments using WebRTC VAD
        
        Returns:
            list: [(start_time, end_time, audio_segment), ...]
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # Process in small frames
        frame_duration = segment_duration  # 30ms frames
        frame_length = int(sr * frame_duration)
        
        speech_frames = []
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            
            # Convert to bytes
            frame_bytes = frame.tobytes()
            
            # Check if speech
            try:
                is_speech = self.vad.is_speech(frame_bytes, sr)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)
        
        # Group consecutive speech frames into segments
        segments = []
        current_start = None
        
        for i, is_speech in enumerate(speech_frames):
            time_pos = i * frame_duration
            
            if is_speech and current_start is None:
                current_start = time_pos
            elif not is_speech and current_start is not None:
                # End of speech segment
                start_idx = int(current_start * sr)
                end_idx = int(time_pos * sr)
                
                if end_idx - start_idx > sr * 0.5:  # At least 0.5 seconds
                    segments.append((
                        current_start,
                        time_pos,
                        audio[start_idx:end_idx]
                    ))
                current_start = None
        
        return segments
    
    def cluster_with_pyannote(self, audio_path):
        """Use pyannote.audio for clustering"""
        try:
            diarization = self.pipeline(audio_path)
            
            results = {
                "method": "pyannote_diarization",
                "total_speakers": len(diarization.labels()),
                "segments": [],
                "speaker_stats": {}
            }
            
            speaker_times = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "speaker_id": f"Speaker_{speaker}",
                    "start_time": str(timedelta(seconds=turn.start)),
                    "end_time": str(timedelta(seconds=turn.end)),
                    "duration": turn.end - turn.start,
                    "confidence": 0.9  # pyannote doesn't provide confidence
                }
                results["segments"].append(segment)
                
                # Track speaker stats
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += segment["duration"]
            
            # Calculate speaker statistics
            total_time = sum(speaker_times.values())
            for speaker, time in speaker_times.items():
                speaker_id = f"Speaker_{speaker}"
                results["speaker_stats"][speaker_id] = {
                    "total_time": str(timedelta(seconds=time)),
                    "percentage": round((time / total_time) * 100, 1) if total_time > 0 else 0
                }
            
            return results
            
        except Exception as e:
            print(f"Pyannote clustering failed: {e}")
            return None
    
    def cluster_with_basic_method(self, speech_segments, audio, sr):
        """Fallback clustering method using MFCC + scikit-learn"""
        if len(speech_segments) < 2:
            return {
                "method": "basic_mfcc",
                "total_speakers": 1,
                "segments": [],
                "speaker_stats": {"Speaker_0": {"total_time": "00:00:00", "percentage": 100}}
            }
        
        # Extract MFCC features for each segment
        embeddings = []
        valid_segments = []
        
        for start_time, end_time, segment_audio in speech_segments:
            if len(segment_audio) > sr * 0.5:  # At least 0.5 seconds
                mfcc = librosa.feature.mfcc(
                    y=segment_audio, 
                    sr=sr, 
                    n_mfcc=13
                )
                mfcc_mean = np.mean(mfcc, axis=1)
                embeddings.append(mfcc_mean)
                valid_segments.append((start_time, end_time, segment_audio))
        
        if len(embeddings) < 2:
            return {
                "method": "basic_mfcc",
                "total_speakers": 1,
                "segments": [],
                "speaker_stats": {"Speaker_0": {"total_time": "00:00:00", "percentage": 100}}
            }
        
        # Determine optimal number of clusters
        embeddings_array = np.array(embeddings)
        n_clusters = self.estimate_speakers(embeddings_array)
        
        # Perform clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        speaker_labels = clusterer.fit_predict(embeddings_array)
        
        # Build results
        results = {
            "method": "basic_mfcc",
            "total_speakers": n_clusters,
            "segments": [],
            "speaker_stats": {}
        }
        
        speaker_times = {}
        
        for i, ((start_time, end_time, _), speaker_id) in enumerate(zip(valid_segments, speaker_labels)):
            duration = end_time - start_time
            speaker_label = f"Speaker_{speaker_id}"
            
            segment = {
                "speaker_id": speaker_label,
                "start_time": str(timedelta(seconds=start_time)),
                "end_time": str(timedelta(seconds=end_time)),
                "duration": duration,
                "confidence": 0.7  # Approximate confidence
            }
            results["segments"].append(segment)
            
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = 0
            speaker_times[speaker_id] += duration
        
        # Calculate speaker statistics
        total_time = sum(speaker_times.values())
        for speaker_id, time in speaker_times.items():
            speaker_label = f"Speaker_{speaker_id}"
            results["speaker_stats"][speaker_label] = {
                "total_time": str(timedelta(seconds=time)),
                "percentage": round((time / total_time) * 100, 1) if total_time > 0 else 0
            }
        
        return results
    
    def estimate_speakers(self, embeddings, max_speakers=8):
        """Estimate optimal number of speakers using silhouette score"""
        if len(embeddings) < 4:
            return min(2, len(embeddings))
        
        max_k = min(max_speakers, len(embeddings) // 2)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_k + 1):
            try:
                clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = clusterer.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        print(f"Estimated {best_k} speakers (silhouette score: {best_score:.3f})")
        return best_k
    
    def save_results(self, results, output_dir, filename):
        """Save clustering results to JSON file"""
        output_file = Path(output_dir) / f"{filename}_clustering_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
        # Print summary
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print clustering summary"""
        print("\n" + "="*50)
        print("VOICE CLUSTERING RESULTS")
        print("="*50)
        print(f"Method: {results['method']}")
        print(f"Total Speakers Detected: {results['total_speakers']}")
        
        if results.get('speaker_stats'):
            print("\nSpeaking Time Distribution:")
            print("-" * 30)
            for speaker, stats in results['speaker_stats'].items():
                print(f"{speaker}: {stats['total_time']} ({stats['percentage']}%)")
        
        print(f"\nTotal Segments: {len(results.get('segments', []))}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Voice Clustering System')
    parser.add_argument('--input', '-i', required=True,
                       help='Input audio file path')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Initialize clustering system
    system = VoiceClusteringSystem()
    
    # Process the audio file
    results = system.process_audio_file(args.input, args.output)
    
    if results:
        print(f"\nProcessing completed successfully!")
        print(f"Found {results['total_speakers']} speakers")
    else:
        print("Processing failed!")

if __name__ == "__main__":
    main()