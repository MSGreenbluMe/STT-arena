"""
STT Arena - Main Streamlit Application
Compare multiple STT providers with Gemini-powered Golden Transcript generation
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import time

from utils import (
    transcribe_with_all_providers,
    GeminiGoldenTranscript,
    calculate_wer_scores,
    format_metadata_for_display
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="STT Arena - Model Comparison",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .golden-transcript {
        background-color: #fff9e6;
        border-left: 5px solid #ffd700;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .provider-transcript {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'transcription_results' not in st.session_state:
        st.session_state.transcription_results = None
    if 'golden_transcript' not in st.session_state:
        st.session_state.golden_transcript = None
    if 'wer_scores' not in st.session_state:
        st.session_state.wer_scores = None
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None


def sidebar_config():
    """Sidebar configuration and API key inputs"""
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Keys")

    # API Key inputs (with environment variable fallback)
    # Try multiple environment variable names for compatibility
    api_keys = {}

    api_keys['gemini'] = st.sidebar.text_input(
        "Google Gemini API Key",
        value=os.getenv('GEMINI_API_KEY', os.getenv('gemini', '')),
        type="password",
        help="Required for Golden Transcript generation"
    )

    api_keys['elevenlabs'] = st.sidebar.text_input(
        "ElevenLabs API Key",
        value=os.getenv('ELEVENLABS_API_KEY', os.getenv('elevenlabs', '')),
        type="password"
    )

    api_keys['gladia'] = st.sidebar.text_input(
        "Gladia API Key",
        value=os.getenv('GLADIA_API_KEY', os.getenv('gladia', '')),
        type="password"
    )

    api_keys['openai'] = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.getenv('OPENAI_API_KEY', os.getenv('openai', '')),
        type="password"
    )

    api_keys['behavioral'] = st.sidebar.text_input(
        "Behavioral Signals API Key",
        value=os.getenv('BEHAVIORAL_SIGNALS_API_KEY', os.getenv('Behavioral signals', '')),
        type="password"
    )

    api_keys['behavioral_url'] = st.sidebar.text_input(
        "Behavioral Signals URL",
        value=os.getenv('BEHAVIORAL_SIGNALS_URL', 'https://api.behavioralsignals.com/v1/transcribe'),
        help="API endpoint for Behavioral Signals"
    )

    api_keys['deepgram'] = st.sidebar.text_input(
        "Deepgram API Key",
        value=os.getenv('DEEPGRAM_API_KEY', os.getenv('Deepgram', '')),
        type="password"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Select STT Providers")

    # Provider selection
    enabled_providers = {}
    enabled_providers['elevenlabs'] = st.sidebar.checkbox(
        "ElevenLabs",
        value=bool(api_keys['elevenlabs']),
        disabled=not bool(api_keys['elevenlabs'])
    )
    enabled_providers['gladia'] = st.sidebar.checkbox(
        "Gladia",
        value=bool(api_keys['gladia']),
        disabled=not bool(api_keys['gladia'])
    )
    enabled_providers['openai'] = st.sidebar.checkbox(
        "OpenAI Whisper",
        value=bool(api_keys['openai']),
        disabled=not bool(api_keys['openai'])
    )
    enabled_providers['behavioral'] = st.sidebar.checkbox(
        "Behavioral Signals",
        value=bool(api_keys['behavioral']),
        disabled=not bool(api_keys['behavioral'])
    )
    enabled_providers['deepgram'] = st.sidebar.checkbox(
        "Deepgram",
        value=bool(api_keys['deepgram']),
        disabled=not bool(api_keys['deepgram'])
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Audio Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio File",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload a call center audio recording"
    )

    return api_keys, enabled_providers, uploaded_file


def main():
    """Main application logic"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🎙️ STT Arena</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Compare Speech-to-Text models with AI-powered Golden Transcript</div>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    api_keys, enabled_providers, uploaded_file = sidebar_config()

    # Check if Gemini API key is provided
    if not api_keys['gemini']:
        st.warning("⚠️ Please provide a Google Gemini API Key in the sidebar to generate Golden Transcripts.")
        st.info("💡 Tip: Create a `.env` file with your API keys or enter them in the sidebar.")
        st.stop()

    # Check if at least one provider is enabled
    if not any(enabled_providers.values()):
        st.info("👈 Please select at least one STT provider from the sidebar and upload an audio file.")
        st.stop()

    # Audio file handling
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_file_path = tmp_file.name
            st.session_state.audio_file_path = audio_file_path

        # Transcribe button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Transcription Arena", use_container_width=True, type="primary"):
                with st.spinner("🔄 Transcribing with multiple providers..."):
                    # Transcribe with all enabled providers
                    results = transcribe_with_all_providers(
                        audio_file_path,
                        enabled_providers,
                        api_keys
                    )

                    st.session_state.transcription_results = results

                    # Generate Golden Transcript
                    if results:
                        with st.spinner("✨ Generating Golden Transcript with Gemini..."):
                            # Extract successful transcripts
                            transcripts = {
                                provider: data['transcript']
                                for provider, data in results.items()
                                if data.get('success') and data.get('transcript')
                            }

                            if transcripts:
                                gemini = GeminiGoldenTranscript(api_keys['gemini'])
                                golden_transcript, metadata = gemini.generate_golden_transcript(transcripts)
                                st.session_state.golden_transcript = golden_transcript

                                # Calculate WER scores
                                wer_scores = calculate_wer_scores(transcripts, golden_transcript)
                                st.session_state.wer_scores = wer_scores
                            else:
                                st.error("❌ No successful transcriptions to generate Golden Transcript.")

        # Display results
        if st.session_state.transcription_results:
            st.markdown("---")
            st.markdown("## 🏆 The Arena Results")

            # Golden Transcript Section
            if st.session_state.golden_transcript:
                st.markdown("### ✨ Golden Transcript (Source of Truth)")
                st.markdown('<div class="golden-transcript">', unsafe_allow_html=True)

                # Editable golden transcript
                edited_golden = st.text_area(
                    "You can edit the Golden Transcript if needed:",
                    value=st.session_state.golden_transcript,
                    height=200,
                    key="golden_edit"
                )

                if edited_golden != st.session_state.golden_transcript:
                    if st.button("🔄 Recalculate WER with Edited Transcript"):
                        st.session_state.golden_transcript = edited_golden
                        # Recalculate WER
                        transcripts = {
                            provider: data['transcript']
                            for provider, data in st.session_state.transcription_results.items()
                            if data.get('success') and data.get('transcript')
                        }
                        st.session_state.wer_scores = calculate_wer_scores(
                            transcripts,
                            edited_golden
                        )
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            # Performance Metrics
            st.markdown("### 📊 Performance Metrics")

            metrics_data = []
            for provider, result in st.session_state.transcription_results.items():
                if result.get('success'):
                    wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                    wer_display = f"{wer_score}%" if wer_score != 'N/A' and wer_score is not None else 'N/A'

                    metrics_data.append({
                        'Provider': provider,
                        'WER Score': wer_display,
                        'Processing Time (s)': f"{result.get('processing_time', 0):.2f}",
                        'Status': '✅ Success'
                    })
                else:
                    metrics_data.append({
                        'Provider': provider,
                        'WER Score': 'N/A',
                        'Processing Time (s)': 'N/A',
                        'Status': f"❌ {result.get('error', 'Failed')}"
                    })

            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Transcript Comparison
            st.markdown("### 📝 Transcript Comparison")

            # Create tabs for each provider
            provider_names = list(st.session_state.transcription_results.keys())
            tabs = st.tabs(provider_names)

            for tab, provider in zip(tabs, provider_names):
                with tab:
                    result = st.session_state.transcription_results[provider]

                    if result.get('success'):
                        st.markdown(f'<div class="provider-transcript">', unsafe_allow_html=True)
                        st.markdown(f"**Transcript:**")
                        st.write(result.get('transcript', 'No transcript available'))
                        st.markdown('</div>', unsafe_allow_html=True)

                        # WER Score
                        wer_score = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                        if wer_score != 'N/A' and wer_score is not None:
                            st.metric("Word Error Rate", f"{wer_score}%")

                        # Metadata
                        with st.expander("🔍 View Metadata & Details"):
                            st.json(result.get('metadata', {}))
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        st.markdown('</div>', unsafe_allow_html=True)

            # Download section
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📥 Download Golden Transcript"):
                    st.download_button(
                        label="Download as TXT",
                        data=st.session_state.golden_transcript,
                        file_name="golden_transcript.txt",
                        mime="text/plain"
                    )

            with col2:
                if st.button("📥 Download All Transcripts"):
                    all_transcripts = "=" * 60 + "\n"
                    all_transcripts += "STT ARENA - ALL TRANSCRIPTS\n"
                    all_transcripts += "=" * 60 + "\n\n"

                    all_transcripts += "GOLDEN TRANSCRIPT:\n"
                    all_transcripts += "-" * 60 + "\n"
                    all_transcripts += st.session_state.golden_transcript + "\n\n"

                    for provider, result in st.session_state.transcription_results.items():
                        if result.get('success'):
                            all_transcripts += f"\n{provider.upper()}:\n"
                            all_transcripts += "-" * 60 + "\n"
                            all_transcripts += result.get('transcript', '') + "\n"
                            wer = st.session_state.wer_scores.get(provider, 'N/A') if st.session_state.wer_scores else 'N/A'
                            all_transcripts += f"WER: {wer}%\n" if wer != 'N/A' else "WER: N/A\n"

                    st.download_button(
                        label="Download All as TXT",
                        data=all_transcripts,
                        file_name="all_transcripts.txt",
                        mime="text/plain"
                    )

    else:
        st.info("👈 Please upload an audio file from the sidebar to begin.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "STT Arena | Powered by Google Gemini | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
