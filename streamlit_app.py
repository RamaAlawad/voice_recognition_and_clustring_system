# File: streamlit_app.py
"""
Streamlit Interface for Voice Recognition System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.audio_processor import AudioProcessor
from src.voice_recognizer import VoiceRecognizer
from src.evaluator import RecognitionEvaluator
from src.data_manager import DataManager

# Page configuration
st.set_page_config(
    page_title="Voice Recognition System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'recognition_results' not in st.session_state:
        st.session_state.recognition_results = None

def initialize_components():
    """Initialize system components"""
    if 'components' not in st.session_state:
        st.session_state.components = {
            'data_manager': DataManager(),
            'audio_processor': AudioProcessor(),
            'recognizer': VoiceRecognizer(),
            'evaluator': RecognitionEvaluator()
        }

def main():
    """Main Streamlit application"""
    
    # Initialize
    init_session_state()
    initialize_components()
    
    # Header
    st.title("🎤 Voice Recognition System")
    st.markdown("**Train on 3 known speakers, then recognize new voices**")
    st.markdown("---")
    
    # Sidebar Navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📚 Training Data", "🎯 Model Training", "🔍 Voice Recognition", "📊 Results & Analysis"]
        )
        
        st.markdown("---")
        
        # System Status
        st.header("📊 System Status")
        
        # Check training data
        data_manager = st.session_state.components['data_manager']
        training_info = data_manager.check_training_data()
        
        if training_info['exists'] and training_info['ready_for_training']:
            st.success("✅ Training data ready")
            st.write(f"**Total files:** {training_info['total_files']}")
            for person, info in training_info['persons'].items():
                st.write(f"• {person}: {info['files_count']} files")
        else:
            st.warning("⚠️ Training data not found")
        
        # Model status
        if st.session_state.training_completed:
            st.success("✅ Model trained")
        else:
            st.info("🔄 Model not trained")
        
        st.markdown("---")
        st.markdown("**Expected Structure:**")
        st.code("""LibriSpeech/test-clean/
├── person1/ (audio files)
├── person2/ (audio files) 
└── person3/ (audio files)""")
    
    # Page routing
    page_key = page.split()[1]  # Extract key from emoji + text
    
    if page_key == "Home":
        show_home_page()
    elif page_key == "Training":
        show_training_data_page()
    elif page_key == "Model":
        show_model_training_page()
    elif page_key == "Voice":
        show_recognition_page()
    elif page_key == "Results":
        show_results_page()

def show_home_page():
    """Home page with system overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏠 Welcome to Voice Recognition System")
        
        st.markdown("""
        ### How This System Works:
        
        **Phase 1: Training** 🎯
        1. **Prepare Data**: Organize voice samples for 3 known persons
        2. **Process Audio**: Extract MFCC features from training voices  
        3. **Train Model**: Build classifier to recognize these 3 speakers
        
        **Phase 2: Recognition** 🔍  
        1. **Upload Voice**: Upload new/unknown voice sample
        2. **Predict Speaker**: System identifies if it's one of the 3 trained speakers
        3. **Get Confidence**: See confidence scores for each possible speaker
        
        ### Your Training Data Structure:
        Based on your VS Code structure, organize files like this:
        """)
        
        st.code("""
        LibriSpeech/test-clean/
        ├── person1/
        │   ├── x.flac
        │   └── (other audio files)
        ├── person2/  
        │   ├── y.flac
        │   └── (other audio files)
        └── person3/
            ├── z.flac  
            └── (other audio files)
        """)
        
        st.markdown("### What You Can Do:")
        
        st.info("📚 **Training Data**: Check your current training data status")
        st.info("🎯 **Model Training**: Train the recognition model on your 3 speakers")  
        st.info("🔍 **Voice Recognition**: Upload new voices to test recognition")
        st.info("📊 **Results**: Analyze recognition performance and confidence")
    
    with col2:
        st.header("🔄 System Workflow")
        
        # Workflow diagram
        workflow_steps = [
            {"step": "1. Load Training Data", "status": "✅" if check_training_data_exists() else "⏳"},
            {"step": "2. Process Audio", "status": "✅" if check_processed_data_exists() else "⏳"},
            {"step": "3. Train Model", "status": "✅" if st.session_state.training_completed else "⏳"},
            {"step": "4. Upload Test Voice", "status": "⏳"},
            {"step": "5. Recognize Speaker", "status": "⏳"}
        ]
        
        for item in workflow_steps:
            st.markdown(f"{item['status']} {item['step']}")
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🚀 Start Training", type="primary"):
            st.switch_page("🎯 Model Training")
        
        if st.button("🎤 Test Recognition"):
            st.rerun()

def show_training_data_page():
    """Training data management page"""
    
    st.header("📚 Training Data Management")
    
    # Get training data info
    data_manager = st.session_state.components['data_manager']
    training_info = data_manager.check_training_data()
    
    if not training_info['exists']:
        st.error("❌ Training data directory not found!")
        st.markdown("""
        **Please create the following directory structure:**
        
        ```
        LibriSpeech/test-clean/
        ├── person1/  (place audio files for person 1)
        ├── person2/  (place audio files for person 2)
        └── person3/  (place audio files for person 3)
        ```
        
        **Supported formats:** FLAC, WAV, MP3
        """)
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Current Training Data")
        
        if training_info['ready_for_training']:
            st.success(f"✅ Ready for training! Total files: {training_info['total_files']}")
            
            # Show data summary
            data_summary = []
            for person, info in training_info['persons'].items():
                data_summary.append({
                    'Person': person,
                    'Audio Files': info['files_count'],
                    'Sample Files': ', '.join(info['files'][:3]) + ('...' if len(info['files']) > 3 else '')
                })
            
            df = pd.DataFrame(data_summary)
            st.dataframe(df, use_container_width=True)
            
            # Visualize distribution
            persons = list(training_info['persons'].keys())
            counts = [training_info['persons'][p]['files_count'] for p in persons]
            
            fig = px.bar(x=persons, y=counts, 
                        title="Audio Files per Person",
                        labels={'x': 'Person', 'y': 'Number of Files'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ No audio files found in training directories")
    
    with col2:
        st.subheader("⚙️ Data Processing Status")
        
        # Check if data is already processed
        existing_data = data_manager.load_training_data()
        
        if existing_data:
            st.success("✅ Training data already processed!")
            st.write(f"**Processed files:** {existing_data['metadata']['n_files']}")
            st.write(f"**Speakers:** {existing_data['metadata']['n_speakers']}")
            st.write(f"**Feature dimensions:** {existing_data['metadata']['feature_dim']}")
            
            if st.button("🔄 Reprocess Data", help="Process data again with new settings"):
                process_training_data()
        else:
            st.info("🔄 Training data not processed yet")
            
            if training_info['ready_for_training']:
                # Processing settings
                st.write("**Processing Settings:**")
                sample_rate = st.selectbox("Sample Rate", [16000, 8000, 22050], index=0)
                n_mfcc = st.slider("MFCC Coefficients", 8, 20, 13)
                
                if st.button("⚡ Process Training Data", type="primary"):
                    with st.spinner("Processing audio files..."):
                        process_training_data(sample_rate, n_mfcc)
                        st.rerun()

def show_model_training_page():
    """Model training page"""
    
    st.header("🎯 Model Training")
    
    # Check if data is processed
    data_manager = st.session_state.components['data_manager']
    existing_data = data_manager.load_training_data()
    
    if not existing_data:
        st.warning("⚠️ Please process training data first!")
        if st.button("📚 Go to Training Data"):
            st.rerun()
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Training Settings")
        
        st.write("**Training Data Info:**")
        st.write(f"• Files: {existing_data['metadata']['n_files']}")
        st.write(f"• Speakers: {existing_data['metadata']['n_speakers']}")
        st.write(f"• Features: {existing_data['metadata']['feature_dim']}")
        
        # Model settings
        model_type = st.selectbox(
            "Model Type",
            ["random_forest", "svm", "knn"],
            help="Random Forest usually works best for voice recognition"
        )
        
        # Training button
        train_button = st.button("🚀 Start Training", type="primary")
        
        if st.session_state.training_completed:
            st.success("✅ Model already trained!")
            if st.button("🔄 Retrain Model"):
                st.session_state.training_completed = False
                st.session_state.trained_model = None
                st.rerun()
    
    with col2:
        st.subheader("📈 Training Results")
        
        if train_button:
            with st.spinner("Training model..."):
                train_model(model_type, existing_data)
                st.rerun()
        
        elif st.session_state.training_completed and st.session_state.trained_model:
            # Show training results
            results = st.session_state.trained_model['results']
            
            # Overall metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Training Accuracy", f"{results['accuracy']:.1%}")
            with col_b:
                st.metric("Model Type", model_type.upper())
            with col_c:
                st.metric("Speakers", len(results['speakers']))
            
            # Per-speaker performance
            st.write("**Per-Speaker Performance:**")
            perf_data = []
            for speaker, metrics in results['per_speaker_metrics'].items():
                perf_data.append({
                    'Speaker': speaker,
                    'Precision': f"{metrics['precision']:.2%}",
                    'Recall': f"{metrics['recall']:.2%}", 
                    'F1-Score': f"{metrics['f1_score']:.2%}",
                    'Samples': metrics['support']
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Confusion Matrix
            st.write("**Confusion Matrix:**")
            cm = results['confusion_matrix']
            speakers = results['speakers']
            
            fig = px.imshow(cm, 
                           x=speakers, y=speakers,
                           aspect="auto", 
                           color_continuous_scale="Blues",
                           text_auto=True)
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Speaker",
                yaxis_title="True Speaker"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_recognition_page():
    """Voice recognition testing page"""
    
    st.header("🔍 Voice Recognition Testing")
    
    if not st.session_state.training_completed:
        st.warning("⚠️ Please train the model first!")
        if st.button("🎯 Go to Training"):
            st.rerun()
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎤 Upload Voice Sample")
        
        st.write("**Known Speakers:**")
        if st.session_state.trained_model:
            speakers = st.session_state.trained_model['results']['speakers']
            for speaker in speakers:
                st.write(f"• {speaker}")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file to test",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload a voice sample to see if it matches any known speaker"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            # Recognition button
            if st.button("🎯 Recognize Speaker", type="primary"):
                with st.spinner("Processing audio and recognizing speaker..."):
                    result = process_recognition(uploaded_file)
                    if result:
                        st.session_state.recognition_results = result
                        st.rerun()
        
        # Show uploaded test files
        st.subheader("📁 Previous Test Files")
        data_manager = st.session_state.components['data_manager']
        uploaded_files = data_manager.get_uploaded_files()
        
        if uploaded_files:
            for file_info in uploaded_files[-5:]:  # Show last 5
                st.write(f"• {file_info['name']} ({file_info['size_mb']:.1f} MB)")
    
    with col2:
        st.subheader("🎯 Recognition Results")
        
        if st.session_state.recognition_results:
            result = st.session_state.recognition_results
            
            # Main result
            st.success(f"🎯 **Predicted Speaker:** {result['predicted_speaker']}")
            
            # Confidence scores
            st.write("**Confidence Scores:**")
            
            conf_data = []
            for speaker, confidence in result['confidence'].items():
                conf_data.append({
                    'Speaker': speaker,
                    'Confidence': confidence,
                    'Percentage': f"{confidence:.1%}"
                })
            
            conf_df = pd.DataFrame(conf_data).sort_values('Confidence', ascending=False)
            st.dataframe(conf_df, use_container_width=True)
            
            # Confidence visualization
            fig = px.bar(conf_df, x='Speaker', y='Confidence',
                        title="Recognition Confidence Scores",
                        text='Percentage')
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            max_conf = max(result['confidence'].values())
            if max_conf > 0.8:
                st.success("🟢 **High Confidence**: Very likely this speaker")
            elif max_conf > 0.6:
                st.warning("🟡 **Medium Confidence**: Possibly this speaker") 
            else:
                st.error("🔴 **Low Confidence**: Might be unknown speaker")
        
        else:
            st.info("Upload an audio file to see recognition results")

def show_results_page():
    """Results analysis page"""
    
    st.header("📊 Results & Analysis")
    
    if not st.session_state.training_completed:
        st.warning("⚠️ No trained model available for analysis")
        return
    
    tab1, tab2, tab3 = st.tabs(["🎯 Training Performance", "🔍 Recognition History", "📈 System Analysis"])
    
    with tab1:
        st.subheader("Training Model Performance")
        
        if st.session_state.trained_model:
            results = st.session_state.trained_model['results']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Accuracy", f"{results['accuracy']:.1%}")
            with col2:
                st.metric("Number of Speakers", len(results['speakers']))
            with col3:
                avg_f1 = np.mean([m['f1_score'] for m in results['per_speaker_metrics'].values()])
                st.metric("Average F1-Score", f"{avg_f1:.1%}")
            with col4:
                total_samples = sum([m['support'] for m in results['per_speaker_metrics'].values()])
                st.metric("Training Samples", total_samples)
            
            # Detailed per-speaker analysis
            st.subheader("Per-Speaker Detailed Analysis")
            
            speakers = results['speakers']
            metrics_data = []
            
            for speaker in speakers:
                metrics = results['per_speaker_metrics'][speaker]
                metrics_data.append({
                    'Speaker': speaker,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'Training Samples': metrics['support']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Bar chart of F1 scores
            fig = px.bar(metrics_df, x='Speaker', y='F1-Score',
                        title="F1-Score by Speaker",
                        text='F1-Score')
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            fig2 = go.Figure()
            
            for metric in ['Precision', 'Recall', 'F1-Score']:
                fig2.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df['Speaker'],
                    y=metrics_df[metric],
                    text=[f"{v:.2%}" for v in metrics_df[metric]],
                    textposition='outside'
                ))
            
            fig2.update_layout(
                title="Performance Metrics Comparison",
                barmode='group',
                yaxis_title="Score"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Recognition History")
        
        # Show previous recognition results
        data_manager = st.session_state.components['data_manager']
        uploaded_files = data_manager.get_uploaded_files()
        
        if uploaded_files:
            st.write(f"Found {len(uploaded_files)} test files:")
            
            files_df = pd.DataFrame(uploaded_files)
            st.dataframe(files_df, use_container_width=True)
            
            # If we have current recognition results
            if st.session_state.recognition_results:
                st.subheader("Latest Recognition Result")
                result = st.session_state.recognition_results
                
                # Show detailed analysis
                st.write(f"**File:** {result.get('filename', 'Unknown')}")
                st.write(f"**Predicted:** {result['predicted_speaker']}")
                st.write(f"**Top Confidence:** {max(result['confidence'].values()):.1%}")
                
                # Decision boundary analysis
                confidences = list(result['confidence'].values())
                confidences.sort(reverse=True)
                
                if len(confidences) >= 2:
                    margin = confidences[0] - confidences[1]
                    st.write(f"**Decision Margin:** {margin:.1%}")
                    
                    if margin > 0.3:
                        st.success("🟢 Clear decision - high confidence")
                    elif margin > 0.1:
                        st.warning("🟡 Moderate decision - medium confidence")
                    else:
                        st.error("🔴 Uncertain decision - low margin")
        else:
            st.info("No test files uploaded yet")
    
    with tab3:
        st.subheader("System Analysis")
        
        # Model information
        if st.session_state.trained_model:
            st.write("**Current Model Configuration:**")
            model_info = st.session_state.trained_model.get('config', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• Model Type: {model_info.get('model_type', 'Unknown')}")
                st.write(f"• Feature Type: MFCC")
                st.write(f"• Sample Rate: 16kHz")
            with col2:
                st.write(f"• MFCC Coefficients: 13")
                st.write(f"• Training Time: {model_info.get('training_time', 'Unknown')}")
                st.write(f"• Model Size: {model_info.get('model_size', 'Unknown')}")
        
        # System recommendations
        st.subheader("🎯 System Recommendations")
        
        if st.session_state.trained_model:
            results = st.session_state.trained_model['results']
            accuracy = results['accuracy']
            
            if accuracy > 0.9:
                st.success("🟢 **Excellent Performance**: Model is working very well!")
            elif accuracy > 0.8:
                st.warning("🟡 **Good Performance**: Consider adding more training data")
                st.write("**Suggestions:**")
                st.write("• Add more voice samples per speaker")
                st.write("• Ensure audio quality is consistent")
            else:
                st.error("🔴 **Poor Performance**: Model needs improvement")
                st.write("**Recommendations:**")
                st.write("• Check audio quality and format")
                st.write("• Increase training data significantly")
                st.write("• Consider different feature extraction methods")

# Helper Functions
def check_training_data_exists():
    """Check if training data exists"""
    data_manager = st.session_state.components['data_manager']
    training_info = data_manager.check_training_data()
    return training_info['exists'] and training_info['ready_for_training']

def check_processed_data_exists():
    """Check if processed data exists"""
    data_manager = st.session_state.components['data_manager']
    existing_data = data_manager.load_training_data()
    return existing_data is not None

def process_training_data(sample_rate=16000, n_mfcc=13):
    """Process training data"""
    try:
        # Update processor settings
        audio_processor = AudioProcessor(sample_rate=sample_rate, n_mfcc=n_mfcc)
        data_manager = st.session_state.components['data_manager']
        
        # Process data
        features, speaker_labels, file_paths = audio_processor.process_training_data(
            "LibriSpeech/test-clean"
        )
        
        # Save processed data
        data_manager.save_training_data(features, speaker_labels, file_paths)
        
        st.success(f"✅ Processed {len(features)} audio files successfully!")
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")

def train_model(model_type, training_data):
    """Train the recognition model"""
    try:
        # Initialize recognizer with selected type
        recognizer = VoiceRecognizer(model_type=model_type)
        
        # Train model
        features = training_data['features']
        labels = training_data['speaker_labels']
        
        train_accuracy = recognizer.train(features, labels)
        
        # Evaluate
        predictions, confidences = recognizer.predict_multiple(features)
        
        evaluator = st.session_state.components['evaluator']
        results = evaluator.evaluate_training(labels, predictions)
        
        # Save model
        recognizer.save_model("models/voice_recognizer.pkl")
        
        # Update session state
        st.session_state.trained_model = {
            'recognizer': recognizer,
            'results': results,
            'config': {
                'model_type': model_type,
                'training_time': 'Just now',
                'model_size': 'Small'
            }
        }
        st.session_state.training_completed = True
        
        st.success(f"✅ Model trained successfully! Accuracy: {train_accuracy:.1%}")
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")

def process_recognition(uploaded_file):
    """Process uploaded file for recognition"""
    try:
        if not st.session_state.trained_model:
            st.error("No trained model available")
            return None
        
        # Save uploaded file
        data_manager = st.session_state.components['data_manager']
        file_path = data_manager.save_uploaded_file(uploaded_file)
        
        # Process audio
        audio_processor = st.session_state.components['audio_processor']
        features = audio_processor.process_audio_file(file_path)
        
        if features is None:
            st.error("Failed to process audio file")
            return None
        
        # Predict
        recognizer = st.session_state.trained_model['recognizer']
        predicted_speaker, confidence_dict = recognizer.predict_speaker(features)
        
        return {
            'filename': uploaded_file.name,
            'predicted_speaker': predicted_speaker,
            'confidence': confidence_dict
        }
        
    except Exception as e:
        st.error(f"❌ Error processing recognition: {str(e)}")
        return None

if __name__ == "__main__":
    main()